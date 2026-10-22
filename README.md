# coupon-uplift-model

Coupon redemption propensity and uplift model built on the Dunnhumby "The Complete
Journey" dataset (2,469 households, 53 weeks, 27 coupon campaigns), loaded via
[`completejourney_py`](https://github.com/bradleyboehmke/completejourney_py).

See [CLAUDE.md](CLAUDE.md) for the full project design: campaign selection, train/test
split, the uplift method, and project stages.

## Data dictionary

Raw tables are saved as parquet in `data/raw/` by [notebooks/01_data_load.ipynb](notebooks/01_data_load.ipynb)
(gitignored — re-run the notebook to regenerate them).

### `transactions` (1,469,307 rows)

Every item scanned at checkout.

| Column | Type | Description |
|---|---|---|
| household_id | int64 | Household that made the purchase |
| store_id | int64 | Store where the purchase occurred |
| basket_id | int64 | Receipt/basket identifier — groups items bought together |
| product_id | int64 | Product purchased, joins to `products` |
| quantity | int64 | Units purchased |
| sales_value | float64 | Amount paid for the item, after discounts |
| retail_disc | float64 | Discount from a non-coupon promotion (e.g. shelf price cut) |
| coupon_disc | float64 | Discount from a manufacturer or store coupon |
| coupon_match_disc | float64 | Store match of a manufacturer coupon discount |
| week | int64 | Week number of the transaction (1–53) |
| transaction_timestamp | datetime64 | Date/time of the transaction |

### `demographics` (801 rows)

Self-reported household demographics — voluntary, so only ~32% of households are covered.

| Column | Type | Description |
|---|---|---|
| household_id | int64 | Household identifier |
| age | str | Age bracket of the household head |
| income | str | Household income bracket |
| home_ownership | str | Homeowner / renter / unknown |
| marital_status | str | Marital status |
| household_size | str | Number of household members (bucketed) |
| household_comp | str | Household composition (e.g. "2 Adults No Kids") |
| kids_count | str | Number of kids (bucketed) |

### `products` (92,331 rows)

Product catalog.

| Column | Type | Description |
|---|---|---|
| product_id | int64 | Product identifier |
| manufacturer_id | int64 | Manufacturer identifier |
| department | str | Department (e.g. GROCERY, DRUG GM) |
| brand | str | Private label or National brand |
| product_category | str | Category within department |
| product_type | str | Finer-grained product type |
| package_size | str | Package size/unit description |

### `campaigns` (6,589 rows)

Which households were targeted by which campaigns.

| Column | Type | Description |
|---|---|---|
| campaign_id | int64 | Campaign identifier, joins to `campaign_descriptions` |
| household_id | int64 | Household targeted by this campaign |

### `campaign_descriptions` (27 rows)

Metadata for each of the 27 campaigns.

| Column | Type | Description |
|---|---|---|
| campaign_id | int64 | Campaign identifier |
| campaign_type | str | Campaign type: Type A, B, or C |
| start_date | datetime64 | Campaign start date |
| end_date | datetime64 | Campaign end date |

### `promotions` (20,940,529 rows)

Weekly in-store/mailer promotional placement, independent of individual households.

| Column | Type | Description |
|---|---|---|
| product_id | int64 | Product on promotion |
| store_id | int64 | Store where the promotion ran |
| display_location | str | In-store display location code |
| mailer_location | str | Mailer placement code |
| week | int64 | Week number of the promotion |

### `coupons` (116,204 rows)

Which coupons (by UPC) belong to which campaign and cover which products.

| Column | Type | Description |
|---|---|---|
| coupon_upc | int64 | Coupon UPC/barcode |
| product_id | int64 | Product the coupon applies to |
| campaign_id | int64 | Campaign the coupon belongs to |

### `coupon_redemptions` (2,102 rows)

Actual coupon redemption events — only possible for households targeted by a campaign.

| Column | Type | Description |
|---|---|---|
| household_id | int64 | Household that redeemed the coupon |
| coupon_upc | int64 | Coupon redeemed, joins to `coupons` |
| campaign_id | int64 | Campaign the redemption belongs to |
| redemption_date | datetime64 | Date of redemption |
