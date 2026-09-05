"""
Generates a synthetic but realistic customer dataset for the segmentation project.
Customers are drawn from 5 hidden 'archetypes' with noise, so that clustering
has genuine structure to recover (rather than pure random data).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1200

CATEGORIES = ["Electronics", "Fashion", "Home & Living", "Beauty", "Groceries", "Sports"]
REGIONS = ["North", "South", "East", "West"]

# Each archetype: (weight, age_mean, age_sd, income_mean, income_sd,
#                   recency_mean, recency_sd, frequency_mean, frequency_sd,
#                   monetary_mean, monetary_sd, online_ratio_mean, membership_mean)
ARCHETYPES = {
    "High-Value Loyalist": dict(weight=0.18, age=(42, 9), income=(95000, 18000),
                                 recency=(8, 5), frequency=(28, 6), monetary=(3200, 700),
                                 online=(0.55, 0.15), membership=(5.5, 2.0)),
    "Young Digital Shopper": dict(weight=0.22, age=(26, 5), income=(48000, 12000),
                                   recency=(6, 4), frequency=(19, 5), monetary=(950, 300),
                                   online=(0.88, 0.08), membership=(1.5, 1.0)),
    "Bargain Hunter": dict(weight=0.20, age=(38, 11), income=(52000, 14000),
                            recency=(20, 10), frequency=(11, 4), monetary=(420, 150),
                            online=(0.5, 0.2), membership=(3.0, 1.8)),
    "At-Risk / Churning": dict(weight=0.20, age=(48, 13), income=(60000, 20000),
                                recency=(85, 25), frequency=(3, 2), monetary=(260, 140),
                                online=(0.35, 0.2), membership=(4.5, 2.5)),
    "New / Occasional Shopper": dict(weight=0.20, age=(31, 10), income=(45000, 15000),
                                      recency=(35, 15), frequency=(4, 2), monetary=(180, 90),
                                      online=(0.65, 0.2), membership=(0.6, 0.5)),
}

rows = []
names = list(ARCHETYPES.keys())
weights = [ARCHETYPES[n]["weight"] for n in names]
assigned = rng.choice(names, size=N, p=weights)

for i, arche in enumerate(assigned):
    a = ARCHETYPES[arche]
    age = int(np.clip(rng.normal(*a["age"]), 18, 75))
    income = max(15000, rng.normal(*a["income"]))
    recency = max(1, rng.normal(*a["recency"]))
    frequency = max(1, rng.normal(*a["frequency"]))
    monetary = max(20, rng.normal(*a["monetary"]))
    online_ratio = float(np.clip(rng.normal(*a["online"]), 0, 1))
    membership_years = max(0.1, rng.normal(*a["membership"]))
    avg_basket = monetary / max(frequency, 1)
    gender = rng.choice(["Female", "Male", "Non-binary"], p=[0.49, 0.48, 0.03])
    region = rng.choice(REGIONS)
    preferred_category = rng.choice(CATEGORIES)
    rows.append(dict(
        customer_id=f"C{10000 + i}",
        age=age,
        gender=gender,
        region=region,
        annual_income=round(income, 2),
        membership_years=round(membership_years, 2),
        recency_days=round(recency, 1),
        frequency_last_year=round(frequency),
        monetary_last_year=round(monetary, 2),
        avg_basket_value=round(avg_basket, 2),
        online_purchase_ratio=round(online_ratio, 2),
        preferred_category=preferred_category,
        # ground-truth label kept ONLY for validation later, not used in clustering
        _true_segment=arche,
    ))

df = pd.DataFrame(rows)
df.to_csv("/home/claude/customer-segmentation/data/customers.csv", index=False)
print(df.shape)
print(df["_true_segment"].value_counts())
print(df.head())
