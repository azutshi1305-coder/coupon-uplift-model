"""Feature and label construction for the propensity/uplift modelling table.

All functions that touch transactions take a pre-period window that has already
been filtered to end strictly before the campaign's start date — nothing in this
module looks at data from on/after a campaign start date when building features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

LOOKBACK_WEEKS = 16
TREND_RECENT_WEEKS = 4
TREND_FILL_VALUE = 1.0
TREND_CAP_LOWER_QUANTILE = 0.01
TREND_CAP_UPPER_QUANTILE = 0.99
DEMOGRAPHIC_COLUMNS = [
    "age",
    "income",
    "home_ownership",
    "marital_status",
    "household_size",
    "household_comp",
    "kids_count",
]


def slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def get_campaign_window(campaign_descriptions: pd.DataFrame, campaign_id: int, lookback_weeks: int = LOOKBACK_WEEKS):
    """Return (window_start, start_date, end_date) for a campaign.

    window_start..start_date (exclusive) is the pre-period used for features.
    start_date..end_date (inclusive of the end_date calendar day) is the campaign window
    used for outcomes.
    """
    info = campaign_descriptions.set_index("campaign_id").loc[campaign_id]
    start_date = info["start_date"]
    end_date = info["end_date"]
    window_start = start_date - pd.Timedelta(weeks=lookback_weeks)
    return window_start, start_date, end_date


def get_top_departments(transactions: pd.DataFrame, products: pd.DataFrame, n: int = 6) -> list[str]:
    """Rank departments by total sales value across the full transaction history."""
    merged = transactions.merge(products[["product_id", "department"]], on="product_id", how="left")
    ranked = merged.groupby("department")["sales_value"].sum().sort_values(ascending=False)
    return ranked.head(n).index.tolist()


def get_pre_period_transactions(transactions_with_dept: pd.DataFrame, window_start, start_date) -> pd.DataFrame:
    """Transactions strictly inside [window_start, start_date) — the only transactions features may use."""
    window_txn = transactions_with_dept[
        (transactions_with_dept["transaction_timestamp"] >= window_start)
        & (transactions_with_dept["transaction_timestamp"] < start_date)
    ].copy()
    if len(window_txn):
        assert window_txn["transaction_timestamp"].max() < start_date, (
            "Feature window leaked into the campaign period"
        )
    return window_txn


def compute_rfm_features(window_txn: pd.DataFrame, start_date) -> pd.DataFrame:
    """Recency, frequency, monetary, and basket-shape features from the pre-period."""
    g = window_txn.groupby("household_id")
    recency_days = (start_date - g["transaction_timestamp"].max()).dt.days.rename("recency_days")
    frequency_baskets = g["basket_id"].nunique().rename("frequency_baskets")
    monetary_spend = g["sales_value"].sum().rename("monetary_spend")
    total_quantity = g["quantity"].sum()

    avg_basket_value = (monetary_spend / frequency_baskets).rename("avg_basket_value")
    items_per_basket = (total_quantity / frequency_baskets).rename("items_per_basket")

    return pd.concat(
        [recency_days, frequency_baskets, monetary_spend, avg_basket_value, items_per_basket], axis=1
    )


def compute_spend_trend_raw(window_txn: pd.DataFrame, start_date, population: list[int]) -> pd.Series:
    """Ratio of average weekly spend in the most recent weeks vs. the weeks before that.

    NaN when a household has zero spend in the earlier sub-window (undefined trend).
    Left uncapped and unfilled here — the cap bounds are fit separately (typically on
    the training campaigns only) and applied uniformly via apply_trend_cap.
    """
    recent_start = start_date - pd.Timedelta(weeks=TREND_RECENT_WEEKS)
    prior_weeks = LOOKBACK_WEEKS - TREND_RECENT_WEEKS

    recent_spend = (
        window_txn[window_txn["transaction_timestamp"] >= recent_start]
        .groupby("household_id")["sales_value"]
        .sum()
        .reindex(population, fill_value=0.0)
    )
    prior_spend = (
        window_txn[window_txn["transaction_timestamp"] < recent_start]
        .groupby("household_id")["sales_value"]
        .sum()
        .reindex(population, fill_value=0.0)
    )

    recent_weekly = recent_spend / TREND_RECENT_WEEKS
    prior_weekly = prior_spend / prior_weeks
    trend = recent_weekly / prior_weekly.replace(0, np.nan)
    return trend.rename("spend_trend_ratio_raw")


def compute_trend_cap_bounds(
    raw_trend: pd.Series, lower_q: float = TREND_CAP_LOWER_QUANTILE, upper_q: float = TREND_CAP_UPPER_QUANTILE
) -> tuple[float, float]:
    """1st/99th percentile bounds from the defined (non-NaN) raw trend values only."""
    valid = raw_trend.dropna()
    return valid.quantile(lower_q), valid.quantile(upper_q)


def apply_trend_cap(
    raw_trend: pd.Series, lower: float, upper: float, fill_value: float = TREND_FILL_VALUE
) -> pd.DataFrame:
    """Clip to [lower, upper], then fill undefined ratios with fill_value and flag them."""
    trend_missing = raw_trend.isna().astype(int).rename("trend_missing")
    trend_capped = raw_trend.clip(lower, upper).fillna(fill_value).rename("spend_trend_ratio")
    return pd.concat([trend_capped, trend_missing], axis=1)


def compute_discount_features(window_txn: pd.DataFrame, population: list[int]) -> pd.DataFrame:
    """Share of spend that came with a retail discount, and total coupon discount dollars."""
    g = window_txn.groupby("household_id")
    total_spend = g["sales_value"].sum().reindex(population, fill_value=0.0)
    total_retail_disc = g["retail_disc"].sum().reindex(population, fill_value=0.0)
    coupon_disc_total = g["coupon_disc"].sum().reindex(population, fill_value=0.0).rename("coupon_disc_total")

    retail_disc_share = (total_retail_disc / total_spend.replace(0, np.nan)).rename("retail_disc_share")

    return pd.concat([retail_disc_share, coupon_disc_total], axis=1)


def compute_campaign_coupon_spend_share(
    window_txn: pd.DataFrame, coupons: pd.DataFrame, campaign_id: int, population: list[int]
) -> pd.Series:
    """Share of pre-period spend on products covered by this campaign's own coupons."""
    campaign_products = set(coupons.loc[coupons["campaign_id"] == campaign_id, "product_id"])

    total_spend = window_txn.groupby("household_id")["sales_value"].sum().reindex(population, fill_value=0.0)
    campaign_spend = (
        window_txn[window_txn["product_id"].isin(campaign_products)]
        .groupby("household_id")["sales_value"]
        .sum()
        .reindex(population, fill_value=0.0)
    )

    share = campaign_spend / total_spend.replace(0, np.nan)
    return share.rename("campaign_coupon_spend_share")


def compute_department_spend_shares(
    window_txn: pd.DataFrame, top_departments: list[str], population: list[int]
) -> pd.DataFrame:
    """Share of pre-period spend in each of the top-N departments (by overall sales value)."""
    total_spend = window_txn.groupby("household_id")["sales_value"].sum().reindex(population, fill_value=0.0)

    shares = {}
    for dept in top_departments:
        dept_spend = (
            window_txn[window_txn["department"] == dept]
            .groupby("household_id")["sales_value"]
            .sum()
            .reindex(population, fill_value=0.0)
        )
        shares[f"spend_share_{slugify(dept)}"] = dept_spend / total_spend.replace(0, np.nan)

    return pd.DataFrame(shares, index=population)


def compute_past_redemption_count(coupon_redemptions: pd.DataFrame, population: list[int], start_date) -> pd.Series:
    """Count of coupon redemptions (any campaign) strictly before the campaign start date."""
    past = coupon_redemptions[coupon_redemptions["redemption_date"] < start_date]
    counts = past.groupby("household_id").size().reindex(population, fill_value=0)
    return counts.rename("past_redemptions_count")


def compute_campaign_exposure(
    campaigns: pd.DataFrame,
    campaign_descriptions: pd.DataFrame,
    population: list[int],
    campaign_id: int,
    start_date,
) -> pd.DataFrame:
    """Prior campaign exposure: how many earlier campaigns targeted this household, and how many
    other campaigns targeting this household were still active when this campaign started.
    """
    other_campaigns = campaigns[campaigns["campaign_id"] != campaign_id].merge(
        campaign_descriptions[["campaign_id", "start_date", "end_date"]], on="campaign_id", how="left"
    )

    earlier = other_campaigns[other_campaigns["start_date"] < start_date]
    earlier_campaigns_count = (
        earlier.groupby("household_id")["campaign_id"]
        .nunique()
        .reindex(population, fill_value=0)
        .rename("earlier_campaigns_count")
    )

    concurrent = other_campaigns[
        (other_campaigns["start_date"] <= start_date) & (other_campaigns["end_date"] >= start_date)
    ]
    concurrent_campaigns_count = (
        concurrent.groupby("household_id")["campaign_id"]
        .nunique()
        .reindex(population, fill_value=0)
        .rename("concurrent_campaigns_count")
    )

    return pd.concat([earlier_campaigns_count, concurrent_campaigns_count], axis=1)


def build_demographics_features(demographics: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode demographics, with a has_demographics flag joined in later.

    dummy_na=True gives an explicit "missing" column for households that have a
    demographics record but left a specific field (e.g. marital_status) blank —
    distinct from households that have no demographics record at all.
    """
    dummies = pd.get_dummies(
        demographics[DEMOGRAPHIC_COLUMNS], prefix=DEMOGRAPHIC_COLUMNS, dummy_na=True
    )
    return pd.concat([demographics[["household_id"]], dummies], axis=1)


def attach_demographics(features: pd.DataFrame, demo_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join demographics onto the feature table and fill in has_demographics / missing dummies."""
    merged = features.merge(demo_features, on="household_id", how="left")
    dummy_cols = [c for c in demo_features.columns if c != "household_id"]
    merged["has_demographics"] = merged["household_id"].isin(demo_features["household_id"]).astype(int)
    merged[dummy_cols] = merged[dummy_cols].fillna(0).astype(int)
    return merged


def compute_labels(
    transactions: pd.DataFrame,
    campaigns: pd.DataFrame,
    coupon_redemptions: pd.DataFrame,
    coupons: pd.DataFrame,
    population: list[int],
    campaign_id: int,
    start_date,
    end_date,
) -> pd.DataFrame:
    """treated / redeemed / outcome_spend / outcome_coupon_spend for the campaign window.

    The campaign window runs from start_date through the end of end_date's calendar day.
    """
    treated_households = set(campaigns.loc[campaigns["campaign_id"] == campaign_id, "household_id"])
    redeemed_households = set(
        coupon_redemptions.loc[coupon_redemptions["campaign_id"] == campaign_id, "household_id"]
    )
    campaign_products = set(coupons.loc[coupons["campaign_id"] == campaign_id, "product_id"])

    window_end_exclusive = end_date + pd.Timedelta(days=1)
    campaign_txn = transactions[
        (transactions["transaction_timestamp"] >= start_date)
        & (transactions["transaction_timestamp"] < window_end_exclusive)
    ]

    outcome_spend = campaign_txn.groupby("household_id")["sales_value"].sum().reindex(population, fill_value=0.0)
    outcome_coupon_spend = (
        campaign_txn[campaign_txn["product_id"].isin(campaign_products)]
        .groupby("household_id")["sales_value"]
        .sum()
        .reindex(population, fill_value=0.0)
    )

    return pd.DataFrame(
        {
            "household_id": population,
            "treated": [int(h in treated_households) for h in population],
            "redeemed": [int(h in redeemed_households) for h in population],
            "outcome_spend": outcome_spend.values,
            "outcome_coupon_spend": outcome_coupon_spend.values,
        }
    )


def build_campaign_features(
    raw: dict[str, pd.DataFrame],
    transactions_with_dept: pd.DataFrame,
    campaign_id: int,
    top_departments: list[str],
    demo_features: pd.DataFrame,
    lookback_weeks: int = LOOKBACK_WEEKS,
) -> dict:
    """Build everything for a campaign except the spend-trend cap.

    The trend cap bounds are fit once across a chosen set of campaigns (see
    build_model_table) and applied uniformly, so this stage only returns the raw,
    uncapped trend ratio alongside the rest of the features and labels.
    """
    campaign_descriptions = raw["campaign_descriptions"]
    window_start, start_date, end_date = get_campaign_window(campaign_descriptions, campaign_id, lookback_weeks)

    window_txn = get_pre_period_transactions(transactions_with_dept, window_start, start_date)
    population = sorted(window_txn["household_id"].unique())

    raw_trend = compute_spend_trend_raw(window_txn, start_date, population)
    rfm = compute_rfm_features(window_txn, start_date)
    discount = compute_discount_features(window_txn, population)
    coupon_share = compute_campaign_coupon_spend_share(window_txn, raw["coupons"], campaign_id, population)
    dept_shares = compute_department_spend_shares(window_txn, top_departments, population)
    past_redemptions = compute_past_redemption_count(raw["coupon_redemptions"], population, start_date)
    exposure = compute_campaign_exposure(raw["campaigns"], campaign_descriptions, population, campaign_id, start_date)

    features = pd.concat([rfm, discount, coupon_share, dept_shares, past_redemptions, exposure], axis=1)
    features = features.reindex(population).reset_index().rename(columns={"index": "household_id"})

    features = attach_demographics(features, demo_features)

    labels = compute_labels(
        raw["transactions"], raw["campaigns"], raw["coupon_redemptions"], raw["coupons"],
        population, campaign_id, start_date, end_date,
    )

    return {
        "campaign_id": campaign_id,
        "population": population,
        "features": features,
        "raw_trend": raw_trend,
        "labels": labels,
    }


def assemble_campaign_table(campaign_data: dict, trend_lower: float, trend_upper: float) -> pd.DataFrame:
    """Apply the (externally fit) trend cap and join everything into one campaign's table."""
    trend_df = apply_trend_cap(campaign_data["raw_trend"], trend_lower, trend_upper)
    trend_df = trend_df.reset_index().rename(columns={"index": "household_id"})

    table = campaign_data["features"].merge(trend_df, on="household_id", how="left")
    table = table.merge(campaign_data["labels"], on="household_id", how="inner")
    table.insert(0, "campaign_id", campaign_data["campaign_id"])
    return table


def build_model_table(
    raw: dict[str, pd.DataFrame],
    campaign_ids=(8, 13, 18),
    lookback_weeks: int = LOOKBACK_WEEKS,
    trend_cap_fit_campaign_ids=(8, 13),
) -> tuple[pd.DataFrame, list[str], tuple[float, float]]:
    """Build the full (household, campaign) modelling table across all requested campaigns.

    spend_trend_ratio's percentile cap is fit only on trend_cap_fit_campaign_ids (the
    training campaigns, 8 and 13, by default) and then applied to every campaign,
    including the out-of-time test campaign (18) — consistent with treating 18 as held-out
    data that must never influence any statistic used to build the features.
    """
    transactions_with_dept = raw["transactions"].merge(
        raw["products"][["product_id", "department"]], on="product_id", how="left"
    )
    top_departments = get_top_departments(raw["transactions"], raw["products"])
    demo_features = build_demographics_features(raw["demographics"])

    campaign_data = {
        campaign_id: build_campaign_features(
            raw, transactions_with_dept, campaign_id, top_departments, demo_features, lookback_weeks
        )
        for campaign_id in campaign_ids
    }

    fit_trend_values = pd.concat(
        [campaign_data[cid]["raw_trend"] for cid in trend_cap_fit_campaign_ids if cid in campaign_data]
    )
    trend_lower, trend_upper = compute_trend_cap_bounds(fit_trend_values)

    tables = [assemble_campaign_table(campaign_data[cid], trend_lower, trend_upper) for cid in campaign_ids]
    model_table = pd.concat(tables, ignore_index=True)
    return model_table, top_departments, (trend_lower, trend_upper)
