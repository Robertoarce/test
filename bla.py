import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

import spend_allocation.tasks.data_ingestion.usa.names as n
from gamma_anp_core.config.config import Config
from gamma_anp_core.exceptions import CustomError
from gamma_anp_core.tasks.base_task import Task
from gamma_anp_core.utils.datetime import get_calendar_year_from_period_start
from gamma_data_manager.data_manager.data_manager import get_data_manager
from gamma_data_manager.schemas.response_model.input import (
    GeoMasterSchema,
    SellOutOwnSchema,
    TotalSpendSemesterSchema,
    TotalSpendYearSchema,
    TouchpointFactsOwnSchema,
)
from spend_allocation.tasks.data_ingestion.utils import (
    get_cbrx_nbrx_or_trx_total_sum,
    pass_date_column_to_previous_saturday,
)


so_schema = SellOutOwnSchema()

logger = logging.getLogger(__name__)

tp_schema = TouchpointFactsOwnSchema()

gs_schema = GeoMasterSchema()

tot_spend_sem_schema = TotalSpendSemesterSchema()
tot_spend_year_schema = TotalSpendYearSchema()


class FormatDataModel(Task):
    name = "format_data_model"

    def __init__(self, config: Config):
        self.config = config
        self.data_manager_load = get_data_manager(**config.data.clean.load)
        self.data_manager_load_s3 = get_data_manager(**config.data.clean.load_s3)
        self.data_manager_save = get_data_manager(**config.data.formatted.save)
        self.data_manager_total_spend = get_data_manager(
            **config.data.raw.load_spend_tables
        )

    def _load_inputs(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        tv = self.data_manager_load_s3.load("television")
        digital = self.data_manager_load_s3.load("hmg_digital")
        salesforce_calls = self.data_manager_load_s3.load("salesforce_calls")
        salesforce_rgn_lunch_learn = self.data_manager_load_s3.load(
            "salesforce_rgn_lunch_learn"
        )
        salesforce_rgn_speaker_program = self.data_manager_load_s3.load(
            "salesforce_rgn_speaker_program"
        )
        salesforce_rte = self.data_manager_load_s3.load("salesforce_rte")
        salesforce_samples = self.data_manager_load_s3.load("salesforce_samples")
        salesforce_sanofi_speaker_program = self.data_manager_load_s3.load(
            "salesforce_sanofi_speaker_program"
        )
        salesforce_sanofi_lunch_learn = self.data_manager_load_s3.load(
            "salesforce_sanofi_lunch_learn"
        )
        patient_support = self.data_manager_load.load("patient_support")
        prescriptions = self.data_manager_load_s3.load("prescriptions_tp")
        sell_out_own = self.data_manager_load_s3.load("sell_out_own")
        nbrx_funnel = self.data_manager_load_s3.load("prescriptions_so")
        frm = self.data_manager_load_s3.load("frm")
        hcp_geo_potential_mapping = self.data_manager_load.load(
            "hcp_geo_potential_mapping"
        )
        dma_region_mapping = self.data_manager_load.load("dma_region_mapping")
        dm_em_dtc = self.data_manager_load.load("dm_em_dtc")
        dm_em_hcp = self.data_manager_load.load("dm_em_hcp")
        competitor_tv = self.data_manager_load.load("competitor_tv")

        # spend table is available in sharepoint General/Data Ingestion
        total_spend_2021_2022 = self.data_manager_total_spend.load(
            "total_spend_2021_2022"
        )
        tot_spend_sem_schema.validate(total_spend_2021_2022)
        total_spend_2021_2022 = tot_spend_sem_schema.cast(total_spend_2021_2022)

        total_spend_2020 = self.data_manager_total_spend.load("total_spend_2020")
        tot_spend_year_schema.validate(total_spend_2020)
        total_spend_2020 = tot_spend_year_schema.cast(total_spend_2020)

        sales_force = pd.concat(
            [
                salesforce_calls,
                salesforce_samples,
                salesforce_rte,
                salesforce_rgn_lunch_learn,
                salesforce_rgn_speaker_program,
                salesforce_sanofi_lunch_learn,
                salesforce_sanofi_speaker_program,
            ]
        ).drop_duplicates()
        sales_force["period_start"] = pass_date_column_to_previous_saturday(
            sales_force["period_start"]
        )

        return (
            tv,
            digital,
            sales_force,
            patient_support,
            sell_out_own,
            prescriptions,
            frm,
            hcp_geo_potential_mapping,
            dma_region_mapping,
            nbrx_funnel,
            dm_em_dtc,
            dm_em_hcp,
            total_spend_2021_2022,
            total_spend_2020,
            competitor_tv,
        )

    def _save_results(self, sell_out_own: pd.DataFrame, touchpoint_facts: pd.DataFrame):
        self.data_manager_save.save(sell_out_own, table_name="sell_out_own")
        self.data_manager_save.save(touchpoint_facts, table_name="touchpoint_facts")

    def _group_touchpoints(
        self,
        df: pd.DataFrame,
        dma_mapping: pd.DataFrame,
        hcp_column: str,
        data_hcp_level: bool = True,
    ) -> pd.DataFrame:

        if data_hcp_level:
            df = pd.merge(
                df,
                dma_mapping,
                how="left",
                on=[hcp_column, tp_schema.internal_product_code],
            )
            logger.warning(
                f"[NaN Geo Code] Removing {(100 * df[tp_schema.internal_geo_code].isna().sum() / df.shape[0]).round(3)} % rows unmatched geo code"
            )
            df.dropna(
                subset=[tp_schema.internal_geo_code],
                inplace=True,
            )
        else:
            df = pd.merge(
                df,
                dma_mapping,
                how="left",
                on=[
                    tp_schema.internal_geo_code,
                    tp_schema.internal_product_code,
                ],
            ).rename(
                columns={
                    gs_schema.sub_national_code: tp_schema.internal_geo_code,
                    tp_schema.internal_geo_code: "dma",
                }
            )

        if "metric" in df.columns:
            grouping_columns = list(
                set(tp_schema.get_column_names())
                - {tp_schema.value}
                - {tp_schema.campaign_code}
                - {tp_schema.internal_response_geo_code}
            )
            agg_columns = [tp_schema.value]
            if set(df[tp_schema.unit].unique()).issubset({n.F_IMPRESSIONS, n.F_CLICKS}):
                # ASSUMPTION: divide by 2 because we suppose same nb of HCP in each cluster of potential
                df[tp_schema.value] /= 2
            elif list(df[tp_schema.unit].unique()) == ["GRP"]:
                df[tp_schema.value] *= (
                    df["nb_hcp_per_dma_key_cross_sub_national_code"]
                    / df["nb_hcp_per_sub_national_code"]
                )

            else:
                logger.warning(
                    f"Workaround not implement for {df[tp_schema.unit].unique()}"
                )
        else:
            self._check_nan_value_dma(df_sales=df)
            grouping_columns = (
                set(so_schema.get_column_names())
                - {so_schema.value}
                - {so_schema.volume}
            )
            agg_columns = [so_schema.value, so_schema.volume]
        return df.groupby(list(grouping_columns), as_index=False)[agg_columns].sum()

    @staticmethod
    def _check_nan_value_dma(df_sales):
        cct_no_geo = (
            df_sales[[n.F_CCT_ID_HASH, so_schema.internal_geo_code]]
            .drop_duplicates()[so_schema.internal_geo_code]
            .isna()
            .sum()
        )
        for channel_type in [n.F_CBRX, n.F_NBRX]:
            lost_abs = get_cbrx_nbrx_or_trx_total_sum(
                channel_type=channel_type,
                df=df_sales[df_sales[so_schema.internal_geo_code].isna()],
            )

            lost_perc = lost_abs / get_cbrx_nbrx_or_trx_total_sum(
                channel_type=channel_type,
                df=df_sales,
            )
            logger.warning(
                f"{cct_no_geo} CCT_ID_HASHs don't have internal_geo_code, accounting for {lost_abs} = {lost_perc}% of {channel_type}"
            )

    @staticmethod
    def _preprocess_hcp_geo_potential_mapping(
        hcp_geo_potential_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        hcp_geo_potential_mapping.drop(
            [tp_schema.internal_geo_code], axis=1, inplace=True
        )
        hcp_geo_potential_mapping.rename(
            columns={gs_schema.sub_national_code: tp_schema.internal_geo_code},
            inplace=True,
        )
        return hcp_geo_potential_mapping

    @staticmethod
    def filter_dataframe_on_nbrx_and_2021(df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df[so_schema.channel_code].str.contains(n.F_NBRX))
            & (df[so_schema.period_start] >= datetime(2021, 7, 1))
            & (df[so_schema.period_start] < datetime(2023, 1, 1))
        ]

    @staticmethod
    def scale_up_value_and_volume(
        df: pd.DataFrame, scale_factor: float
    ) -> pd.DataFrame:
        if so_schema.volume in df.columns:
            df[so_schema.volume] = df[so_schema.volume] / scale_factor
        df[so_schema.value] = df[so_schema.value] / scale_factor
        return df

    def scale_up_prescriptions_and_nbrx(
        self, sell_out_own, nbrx_funnel, prescriptions
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        so_nbrx_2021 = self.filter_dataframe_on_nbrx_and_2021(sell_out_own)
        so_nbrx_funnel_2021 = self.filter_dataframe_on_nbrx_and_2021(nbrx_funnel)
        scale_factor = (
            so_nbrx_funnel_2021[so_schema.volume].sum()
            / so_nbrx_2021[so_schema.volume].sum()
        )
        logger.warning(
            f"Dividing prescriptions, volume and value NBRx funnel by {scale_factor}"
        )
        return (
            self.scale_up_value_and_volume(nbrx_funnel, scale_factor),
            self.scale_up_value_and_volume(prescriptions, scale_factor),
        )

    @staticmethod
    def preprocess_total_spend(total_spend_table: pd.DataFrame, date: str):
        return total_spend_table[
            [
                date,
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                tot_spend_sem_schema.total_spend,
            ]
        ]

    @staticmethod
    def add_semester(touchpoints: pd.DataFrame):
        touchpoints[n.F_SEMESTER] = "H" + (
            np.where(touchpoints[tp_schema.period_start].dt.quarter.gt(2), 2, 1).astype(
                str
            )
            + touchpoints[tp_schema.period_start].dt.year.apply(str)
        )
        return touchpoints

    @staticmethod
    def calculate_total_exec_per_tactic_indication(
        touchpoints: pd.DataFrame, date: str
    ):
        total_exec = touchpoints.groupby(
            [
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                date,
            ],
            as_index=False,
        ).agg({tp_schema.value: "sum"})

        return total_exec

    @staticmethod
    def calculate_cost_per_exec_per_tactic_indication(
        total_exec: pd.DataFrame,
        total_spend: pd.DataFrame,
        date: str,
    ):
        total_exec_and_spend = total_exec.merge(
            total_spend,
            how="inner",
            on=[
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                date,
            ],
        )
        total_exec_and_spend["cost_per_exec"] = (
            total_exec_and_spend[tot_spend_sem_schema.total_spend]
            / total_exec_and_spend[tp_schema.value]
        )
        cost_per_exec = total_exec_and_spend[
            [
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                date,
                "cost_per_exec",
            ]
        ]
        return cost_per_exec

    def duplicate_some_costs_to_match_touchpoints(self, cost_per_exec: pd.DataFrame):
        # need to duplicate Resp cost into AM and NP to match execution
        cost_per_exec = self.duplicate_resp_to_am_and_np(
            cost_per_exec=cost_per_exec.copy()
        )
        # need to duplicate detailing AM cost into detailing_excl_pulm_sf and detailing_pulm_sf to match execution
        cost_per_exec = self.duplicate_detailing_am_to_excl_pulm_sf_and_pulm_sf(
            cost_per_exec=cost_per_exec
        )
        return cost_per_exec

    @staticmethod
    def duplicate_resp_to_am_and_np(cost_per_exec: pd.DataFrame):
        cost_per_exec.loc[
            cost_per_exec[tp_schema.internal_product_code] == n.F_RESP,
            tp_schema.internal_product_code,
        ] = cost_per_exec.loc[:, tp_schema.internal_product_code].map(
            {n.F_RESP: [n.F_AM, n.F_NP]}
        )

        cost_per_exec = cost_per_exec.explode(
            tp_schema.internal_product_code
        ).reset_index(drop=True)
        return cost_per_exec

    @staticmethod
    def duplicate_detailing_am_to_excl_pulm_sf_and_pulm_sf(
        cost_per_exec: pd.DataFrame,
    ):
        cost_per_exec.loc[
            (cost_per_exec[tp_schema.internal_product_code] == n.F_AM)
            & (cost_per_exec[tp_schema.internal_touchpoint_code] == "detailing"),
            tp_schema.internal_touchpoint_code,
        ] = cost_per_exec.loc[:, tp_schema.internal_touchpoint_code].map(
            {"detailing": ["detailing_excl_pulm_sf", "detailing_pulm_sf"]}
        )

        cost_per_exec = cost_per_exec.explode(
            tp_schema.internal_touchpoint_code
        ).reset_index(drop=True)

        return cost_per_exec

    @staticmethod
    def create_touchpoint_spend(
        touchpoints: pd.DataFrame, cost_per_exec: pd.DataFrame, date: str
    ):

        touchpoints_spend = touchpoints.merge(
            cost_per_exec,
            how="left",
            on=[
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                date,
            ],
        )
        touchpoints_spend[tp_schema.value] *= touchpoints_spend["cost_per_exec"]
        touchpoints_spend[tp_schema.metric] = "spend_value"
        return touchpoints_spend

    @staticmethod
    def add_date(touchpoints: pd.DataFrame, date: str):
        if date == "semester":
            touchpoints[date] = "H" + (
                np.where(
                    touchpoints[tp_schema.period_start].dt.quarter.gt(2), 2, 1
                ).astype(str)
                + touchpoints[tp_schema.period_start].dt.year.apply(str)
            )
        elif date == n.F_YEAR:
            touchpoints[date] = get_calendar_year_from_period_start(
                touchpoints[tp_schema.period_start]
            )
        return touchpoints

    def group_some_exec_to_match_spend(self, total_exec: pd.DataFrame):
        # for tactic 'detailing' and indication 'AM'
        # need to group detailing_excl_pulm_df and detailing_pulm_df executions into detailing AM to match spend
        total_exec = self.group_detailing_am_exec(total_exec=total_exec)
        # need to group some tactics AM and NP executions into Resp to match spend
        total_exec = self.group_resp_exec(total_exec=total_exec)
        return total_exec

    @staticmethod
    def group_resp_exec(total_exec: pd.DataFrame):
        total_exec_resp = total_exec.loc[
            (total_exec[tp_schema.internal_product_code] == n.F_AM)
            | (total_exec[tp_schema.internal_product_code] == n.F_NP)
        ].copy()
        total_exec_resp[tp_schema.internal_product_code] = n.F_RESP

        total_exec_resp = total_exec_resp.groupby(
            [
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                n.F_SEMESTER,
            ],
            as_index=False,
        ).agg({tp_schema.value: "sum"})

        return pd.concat([total_exec, total_exec_resp])

    @staticmethod
    def group_subtactic_to_detailing_am(total_exec: pd.DataFrame, subtactic: str):
        total_exec[tp_schema.internal_touchpoint_code] = np.where(
            total_exec[tp_schema.internal_touchpoint_code] == subtactic,
            "detailing",
            total_exec[tp_schema.internal_touchpoint_code],
        )

        total_exec = total_exec.groupby(
            [
                tp_schema.internal_product_code,
                tp_schema.internal_touchpoint_code,
                n.F_SEMESTER,
            ],
            as_index=False,
        ).agg({tp_schema.value: "sum"})

        return total_exec

    def group_detailing_am_exec(self, total_exec: pd.DataFrame):
        total_exec = self.group_subtactic_to_detailing_am(
            total_exec=total_exec, subtactic="detailing_excl_pulm_sf"
        )
        total_exec = self.group_subtactic_to_detailing_am(
            total_exec=total_exec, subtactic="detailing_pulm_sf"
        )
        return total_exec

    @staticmethod
    def compute_total_spend_yearly(
        total_spend_2020: pd.DataFrame, total_spend_2021_2022: pd.DataFrame
    ) -> pd.DataFrame:
        total_spend_2021_2022["year"] = total_spend_2021_2022["semester"].str[-4:]
        agg_yearly_spend = pd.concat([total_spend_2021_2022, total_spend_2020])
        agg_yearly_spend = agg_yearly_spend.groupby(
            [
                n.F_YEAR,
                tp_schema.internal_touchpoint_code,
                tp_schema.internal_product_code,
            ],
            as_index=False,
        )[tot_spend_sem_schema.total_spend].sum()
        return agg_yearly_spend

    @staticmethod
    def replace_tp_indic_for_check(df):
        df[tp_schema.internal_touchpoint_code] = df[
            tp_schema.internal_touchpoint_code
        ].replace(
            {"detailing_excl_pulm_sf": "detailing", "detailing_pulm_sf": "detailing"}
        )
        df[tp_schema.internal_product_code] = df[
            tp_schema.internal_product_code
        ].replace({"AM": "Resp", "NP": "Resp"})
        return df

    def _check_final_spend_in_tp_facts(
        self,
        yearly_spend_per_ind_tp_initial: pd.DataFrame,
        total_spend_2021_2022: pd.DataFrame,
        total_spend_2020: pd.DataFrame,
    ):
        total_spend_2021_2022[n.F_YEAR] = total_spend_2021_2022[
            tp_schema.period_start
        ].dt.year
        agg_yearly_spend_final = pd.concat([total_spend_2021_2022, total_spend_2020])
        agg_yearly_spend_final = self.replace_tp_indic_for_check(agg_yearly_spend_final)
        yearly_spend_per_ind_tp_initial = self.replace_tp_indic_for_check(
            yearly_spend_per_ind_tp_initial
        )
        agg_yearly_spend_final = agg_yearly_spend_final.groupby(
            [
                n.F_YEAR,
                tp_schema.internal_touchpoint_code,
                tp_schema.internal_product_code,
            ],
            as_index=False,
        )[tp_schema.value].sum()
        yearly_spend_per_ind_tp_initial = yearly_spend_per_ind_tp_initial.groupby(
            [
                n.F_YEAR,
                tp_schema.internal_touchpoint_code,
                tp_schema.internal_product_code,
            ],
            as_index=False,
        )[tot_spend_sem_schema.total_spend].sum()

        compare_df = yearly_spend_per_ind_tp_initial.merge(
            agg_yearly_spend_final,
            on=[
                n.F_YEAR,
                tp_schema.internal_touchpoint_code,
                tp_schema.internal_product_code,
            ],
            how="inner",
        )
        compare_df["check"] = compare_df.total_spend.round() == compare_df.value.round()

        if compare_df["check"].sum() != compare_df.shape[0]:
            raise CustomError(
                "The final spends in the touchpoints facts are not matching the one from the input table."
            )

    def add_spend_in_tp_facts(
        self,
        touchpoints: pd.DataFrame,
        total_spend_2021_2022: pd.DataFrame,
        total_spend_2020: pd.DataFrame,
    ) -> pd.DataFrame:
        yearly_spend_per_ind_tp_initial = self.compute_total_spend_yearly(
            total_spend_2020.copy(), total_spend_2021_2022.copy()
        )

        # 2021 and 2022 spends are given by semester
        touchpoints_spend_2021_2022 = self.create_touchpoints_spend(
            touchpoints=touchpoints.copy(),
            total_spend=total_spend_2021_2022,
            date=n.F_SEMESTER,
        )
        # 2020 spends are given by year
        touchpoints_spend_2020 = self.create_touchpoints_spend(
            touchpoints=touchpoints.copy(),
            total_spend=total_spend_2020,
            date=n.F_YEAR,
        )
        self._check_final_spend_in_tp_facts(
            yearly_spend_per_ind_tp_initial.copy(),
            touchpoints_spend_2021_2022.copy(),
            touchpoints_spend_2020.copy(),
        )
        touchpoints_exec_and_spend = pd.concat(
            [touchpoints, touchpoints_spend_2021_2022, touchpoints_spend_2020]
        ).drop(columns=[n.F_SEMESTER, n.F_YEAR, "cost_per_exec"])

        return touchpoints_exec_and_spend

    def create_touchpoints_spend(
        self, touchpoints: pd.DataFrame, total_spend: pd.DataFrame, date: str
    ):
        total_spend = self.preprocess_total_spend(
            total_spend_table=total_spend, date=date
        )
        touchpoints = self.add_date(touchpoints=touchpoints, date=date)
        total_exec = touchpoints.copy()
        if date == n.F_SEMESTER:
            total_exec = self.group_some_exec_to_match_spend(total_exec=total_exec)
        total_exec = self.calculate_total_exec_per_tactic_indication(
            touchpoints=total_exec, date=date
        )

        cost_per_exec = self.calculate_cost_per_exec_per_tactic_indication(
            total_exec=total_exec,
            total_spend=total_spend,
            date=date,
        )
        if date == n.F_SEMESTER:
            cost_per_exec = self.duplicate_some_costs_to_match_touchpoints(
                cost_per_exec=cost_per_exec
            )
        touchpoints_spend = self.create_touchpoint_spend(
            touchpoints=touchpoints.copy(),
            cost_per_exec=cost_per_exec,
            date=date,
        )
        touchpoints_spend = touchpoints_spend.loc[
            touchpoints_spend[tp_schema.value].notna()
        ]

        return touchpoints_spend

    def aggregate_cleaned_spend(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        agg_cleaned_spend = (
            self.add_date(df, date=date)
            .groupby(
                [
                    date,
                    tp_schema.internal_touchpoint_code,
                    tp_schema.internal_product_code,
                ],
                as_index=False,
            )[["value"]]
            .sum()
            .rename(columns={"value": tot_spend_sem_schema.total_spend})
            .assign(source="Cleaned data")
        )
        return agg_cleaned_spend

    def update_spend_with_cleaned(
        self, total_spend: pd.DataFrame, cleaned_spend: pd.DataFrame, date: str
    ) -> pd.DataFrame:
        merged_spend = total_spend.merge(
            cleaned_spend,
            on=[
                date,
                tp_schema.internal_touchpoint_code,
                tp_schema.internal_product_code,
            ],
            suffixes=["", "_cleaned"],
            how="left",
        )
        merged_spend[tot_spend_sem_schema.total_spend] = np.where(
            merged_spend[f"{tot_spend_sem_schema.total_spend}_cleaned"].notna(),
            merged_spend[f"{tot_spend_sem_schema.total_spend}_cleaned"],
            merged_spend[tot_spend_sem_schema.total_spend],
        )
        merged_spend["source"] = np.where(
            merged_spend[f"{tot_spend_sem_schema.total_spend}_cleaned"].notna(),
            merged_spend["source_cleaned"],
            merged_spend["source"],
        )
        merged_spend.drop(
            columns=[f"{tot_spend_sem_schema.total_spend}_cleaned", "source_cleaned"]
        )
        return merged_spend

    def _update_spend_with_cleaned_data(
        self, total_spend: pd.DataFrame, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Separate exec and spend
        df_spend = df[df.unit == n.F_SPEND].copy()
        df_exec = df[df.unit != n.F_SPEND].copy()

        # Aggregate cleaned spend
        agg_df_spend = self.aggregate_cleaned_spend(df=df_spend, date=n.F_SEMESTER)
        agg_df_spend = agg_df_spend[agg_df_spend[n.F_SEMESTER] == "H22022"].copy()

        # Update total spend with cleaned one
        total_spend = self.update_spend_with_cleaned(
            total_spend=total_spend, cleaned_spend=agg_df_spend, date=n.F_SEMESTER
        )

        return total_spend, df_exec

    def run(self):

        (
            media_grp,
            digital,
            sales_force,
            patient_support,
            sell_out_own,
            prescriptions,
            frm,
            hcp_geo_potential_mapping,
            dma_region_mapping,
            nbrx_funnel,
            dm_em_dtc,
            dm_em_hcp,
            total_spend_2021_2022,
            total_spend_2020,
            competitor_tv,
        ) = self._load_inputs()

        media_grp.internal_geo_code = media_grp.internal_geo_code.astype(float)
        media_grp = media_grp.query("unit=='GRP'")

        total_spend_2021_2022, digital = self._update_spend_with_cleaned_data(
            total_spend=total_spend_2021_2022, df=digital
        )

        hcp_geo_potential_mapping = self._preprocess_hcp_geo_potential_mapping(
            hcp_geo_potential_mapping
        ).drop_duplicates()

        nbrx_funnel, prescriptions = self.scale_up_prescriptions_and_nbrx(
            sell_out_own, nbrx_funnel, prescriptions
        )

        sell_out_data_model = pd.concat(
            [
                self._group_touchpoints(
                    sell_out_own,
                    hcp_geo_potential_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                ),
                self._group_touchpoints(
                    nbrx_funnel,
                    hcp_geo_potential_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                ),
            ]
        )

        touchpoints_data_model = pd.concat(
            [
                self._group_touchpoints(
                    sales_force,
                    hcp_geo_potential_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                ),
                self._group_touchpoints(
                    frm, hcp_geo_potential_mapping, hcp_column=n.F_CCT_ID_HASH
                ),
                self._group_touchpoints(
                    media_grp,
                    dma_region_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                    data_hcp_level=False,
                ),
                self._group_touchpoints(
                    digital,
                    dma_region_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                    data_hcp_level=False,
                ),
                self._group_touchpoints(
                    prescriptions,
                    hcp_geo_potential_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                ),
                self._group_touchpoints(
                    dm_em_dtc,
                    dma_region_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                    data_hcp_level=False,
                ),
                self._group_touchpoints(
                    dm_em_hcp,
                    hcp_geo_potential_mapping,
                    hcp_column=n.F_CCT_ID_HASH,
                ),
                competitor_tv,
            ]
        )
        touchpoints_data_model["internal_response_geo_code"] = "usa"
        # TODO: for now, we put unknown in campaign and unit in currency but this can evolve
        touchpoints_data_model["campaign_code"] = "unknown"

        # Copy units to metrics
        touchpoints_data_model["metric"] = touchpoints_data_model["unit"]

        # Add TRx
        sell_out_data_model = self._add_trx_sell_out_data(
            self._round_sell_out_nbrx(sell_out_data_model)
        )

        # Rounding
        touchpoints_data_model = self._round_tp_prescriptions(touchpoints_data_model)

        # Add total spend
        touchpoints_data_model_with_total_spend = self.add_spend_in_tp_facts(
            touchpoints_data_model, total_spend_2021_2022, total_spend_2020
        )
        self._save_results(
            touchpoint_facts=touchpoints_data_model_with_total_spend,
            sell_out_own=sell_out_data_model,
        )
        return {
            "touchpoint_facts": touchpoints_data_model_with_total_spend,
            "sell_out_own": sell_out_data_model,
        }

    @staticmethod
    def _add_trx_sell_out_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        @param df : sell_out_data_model Dataframe

        @return sell_out_data_model with Trx channel code
        """

        cbrx_nbrx_old_mask = df[so_schema.channel_code].isin(
            [n.F_CBRX, n.F_NBRX + "_old"]
        )  # The Trx value is the sum over channel_code for cbrx and nbrx_old

        group_by_key = [
            so_schema.period_start,
            so_schema.frequency,
            so_schema.internal_response_geo_code,
            so_schema.internal_geo_code,
            so_schema.currency,
            so_schema.internal_product_code,
            so_schema.sale_type,
        ]

        sell_out_data_model_filtered_trx = (
            df[cbrx_nbrx_old_mask]
            .groupby(group_by_key)
            .sum([so_schema.value, so_schema.volume])
            .reset_index()
        )

        sell_out_data_model_filtered_trx[so_schema.channel_code] = n.F_TRX

        sell_out_data_model = pd.concat([df, sell_out_data_model_filtered_trx])

        return sell_out_data_model

    @staticmethod
    def _round_tp_prescriptions(df: pd.DataFrame) -> pd.DataFrame:
        df[tp_schema.value] = np.where(
            df[tp_schema.internal_touchpoint_code] == "prescriptions",
            df[tp_schema.value].round(0),
            df[tp_schema.value],
        )
        return df

    @staticmethod
    def _round_sell_out_nbrx(df: pd.DataFrame) -> pd.DataFrame:
        for col in [so_schema.value, so_schema.volume]:
            df[col] = np.where(
                df[so_schema.channel_code].str.contains(n.F_NBRX),
                df[col].round(0),
                df[col],
            )
        return df
