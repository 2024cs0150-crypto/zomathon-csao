"""
Cart Super Add-On (CSAO) Recommendation System
Synthetic Dataset Generator
================================================
Generates realistic synthetic data mimicking a food delivery platform
for a context-aware Top-K ranking recommendation problem.

Tables Generated:
  1. users_df         - 10,000 users
  2. restaurants_df   - 500 restaurants
  3. menu_items_df    - 4,000 menu items
  4. sessions_df      - 100,000 sessions
  5. cart_events_df   - ~250,000 cart events
  6. model_dataset_df - Final engineered feature set for ranking model
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# GLOBAL SEED & CONSTANTS
# ──────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

N_USERS       = 10_000
N_RESTAURANTS = 500
N_ITEMS       = 4_000
N_SESSIONS    = 100_000
TARGET_EVENTS = 250_000

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune"]

# City → dominant cuisines (realistic Indian food-delivery patterns)
CITY_CUISINE_WEIGHTS = {
    "Mumbai":    {"North Indian": 0.25, "Chinese": 0.20, "Fast Food": 0.20,
                  "South Indian": 0.10, "Biryani": 0.10, "Desserts": 0.08, "Other": 0.07},
    "Delhi":     {"North Indian": 0.35, "Chinese": 0.15, "Fast Food": 0.18,
                  "Biryani": 0.12, "South Indian": 0.08, "Desserts": 0.07, "Other": 0.05},
    "Bengaluru": {"South Indian": 0.30, "North Indian": 0.20, "Chinese": 0.15,
                  "Fast Food": 0.15, "Biryani": 0.10, "Desserts": 0.06, "Other": 0.04},
    "Hyderabad": {"Biryani": 0.35, "South Indian": 0.20, "North Indian": 0.18,
                  "Chinese": 0.10, "Fast Food": 0.10, "Desserts": 0.05, "Other": 0.02},
    "Chennai":   {"South Indian": 0.40, "Biryani": 0.20, "North Indian": 0.15,
                  "Chinese": 0.10, "Fast Food": 0.08, "Desserts": 0.05, "Other": 0.02},
    "Pune":      {"North Indian": 0.28, "Fast Food": 0.22, "Chinese": 0.18,
                  "South Indian": 0.12, "Biryani": 0.10, "Desserts": 0.06, "Other": 0.04},
}
CUISINES = list(list(CITY_CUISINE_WEIGHTS.values())[0].keys())

CATEGORIES = ["main", "beverage", "dessert", "side"]

# Meal-time windows (hour ranges)
MEAL_WINDOWS = {
    "breakfast":  (7,  10),
    "lunch":      (12, 15),
    "dinner":     (19, 22),
    "late-night": (22, 24),
}

# ──────────────────────────────────────────────
# 1. USERS TABLE
# ──────────────────────────────────────────────
print("Generating users...")

segments = rng.choice(["budget", "mid", "premium"],
                      size=N_USERS, p=[0.45, 0.35, 0.20])

# AOV depends on segment
aov_map = {"budget": (150, 50), "mid": (350, 100), "premium": (700, 200)}
avg_order_value = np.array([
    max(80, rng.normal(aov_map[s][0], aov_map[s][1]))
    for s in segments
])

# Order frequency: ~15% cold-start users have very low history
lifetime_orders = np.where(
    rng.random(N_USERS) < 0.15,                        # cold-start
    rng.integers(1, 4, N_USERS),                        # 1-3 orders
    rng.integers(5, 200, N_USERS)                       # regular users
)
order_frequency = np.clip(lifetime_orders / 30, 0.1, 10)  # orders/month proxy

# Assign city first, then derive preferred cuisine from city weights
user_cities = rng.choice(CITIES, size=N_USERS)
preferred_cuisine = np.array([
    rng.choice(
        list(CITY_CUISINE_WEIGHTS[c].keys()),
        p=list(CITY_CUISINE_WEIGHTS[c].values())
    )
    for c in user_cities
])

veg_preference_ratio = np.clip(rng.beta(2, 3, N_USERS), 0, 1)  # slight non-veg lean
price_sensitivity_score = np.where(
    segments == "budget",  rng.uniform(0.6, 1.0, N_USERS),
    np.where(segments == "mid", rng.uniform(0.3, 0.7, N_USERS),
             rng.uniform(0.0, 0.4, N_USERS))
)

# Recency score: higher = more recent activity (0-1)
recency_score = np.clip(
    np.where(lifetime_orders < 4, rng.uniform(0, 0.3, N_USERS),
             rng.beta(3, 2, N_USERS)),
    0, 1
)

users_df = pd.DataFrame({
    "user_id":               np.arange(1, N_USERS + 1),
    "city":                  user_cities,
    "segment":               segments,
    "avg_order_value":       avg_order_value.round(2),
    "order_frequency":       order_frequency.round(3),
    "preferred_cuisine":     preferred_cuisine,
    "veg_preference_ratio":  veg_preference_ratio.round(3),
    "price_sensitivity_score": price_sensitivity_score.round(3),
    "recency_score":         recency_score.round(3),
    "lifetime_orders":       lifetime_orders,
})

print(f"  Users shape: {users_df.shape}")

# ──────────────────────────────────────────────
# 2. RESTAURANTS TABLE
# ──────────────────────────────────────────────
print("Generating restaurants...")

rest_cities = rng.choice(CITIES, size=N_RESTAURANTS)
rest_cuisines = np.array([
    rng.choice(
        list(CITY_CUISINE_WEIGHTS[c].keys()),
        p=list(CITY_CUISINE_WEIGHTS[c].values())
    )
    for c in rest_cities
])

# price_range 1=budget, 2=mid, 3=premium (long-tail: mostly budget/mid)
price_range = rng.choice([1, 2, 3], size=N_RESTAURANTS, p=[0.45, 0.40, 0.15])

# Rating: slightly skewed toward 3.5-4.5 (real-world delivery platforms)
rating = np.clip(rng.normal(4.0, 0.5, N_RESTAURANTS), 1.0, 5.0).round(1)

# Order volume: long-tail distribution (few restaurants dominate)
order_volume = (rng.pareto(1.5, N_RESTAURANTS) * 500 + 50).astype(int)

is_chain = rng.random(N_RESTAURANTS) < 0.30  # 30% chain restaurants

# Peak hour multiplier: chains and higher-rated restaurants spike more
peak_hour_multiplier = np.where(
    is_chain, rng.uniform(1.3, 2.0, N_RESTAURANTS),
    rng.uniform(1.0, 1.5, N_RESTAURANTS)
).round(2)

restaurants_df = pd.DataFrame({
    "restaurant_id":        np.arange(1, N_RESTAURANTS + 1),
    "city":                 rest_cities,
    "cuisine_type":         rest_cuisines,
    "price_range":          price_range,
    "rating":               rating,
    "order_volume":         order_volume,
    "is_chain":             is_chain,
    "peak_hour_multiplier": peak_hour_multiplier,
})

print(f"  Restaurants shape: {restaurants_df.shape}")

# ──────────────────────────────────────────────
# 3. MENU ITEMS TABLE
# ──────────────────────────────────────────────
print("Generating menu items...")

# Distribute items across restaurants (some restaurants have more items)
items_per_restaurant = rng.integers(4, 20, N_RESTAURANTS)
# Normalize to exactly N_ITEMS
items_per_restaurant = np.round(
    items_per_restaurant / items_per_restaurant.sum() * N_ITEMS
).astype(int)
# Fix rounding drift
diff = N_ITEMS - items_per_restaurant.sum()
items_per_restaurant[0] += diff

item_restaurant_ids = np.repeat(np.arange(1, N_RESTAURANTS + 1), items_per_restaurant)

# Category distribution per item (realistic: mains dominate)
item_categories = rng.choice(CATEGORIES, size=N_ITEMS, p=[0.45, 0.25, 0.15, 0.15])

# Price depends on category + restaurant price_range
rest_price_range_map = dict(zip(restaurants_df["restaurant_id"], restaurants_df["price_range"]))
item_prices = []
for cat, rid in zip(item_categories, item_restaurant_ids):
    pr = rest_price_range_map[rid]
    if cat == "main":
        base = {1: (80, 180), 2: (150, 350), 3: (300, 700)}[pr]
    elif cat == "beverage":
        base = {1: (30, 80),  2: (60, 150),  3: (120, 300)}[pr]
    elif cat == "dessert":
        base = {1: (50, 120), 2: (100, 250), 3: (200, 500)}[pr]
    else:  # side
        base = {1: (40, 100), 2: (80, 200),  3: (150, 400)}[pr]
    item_prices.append(round(rng.uniform(base[0], base[1]), 2))

# Veg flag: depends on cuisine (South Indian / Desserts tend to be more veg)
rest_cuisine_map = dict(zip(restaurants_df["restaurant_id"], restaurants_df["cuisine_type"]))
veg_probs = []
for rid in item_restaurant_ids:
    cuisine = rest_cuisine_map[rid]
    p = {"South Indian": 0.80, "Desserts": 0.90, "North Indian": 0.55,
         "Biryani": 0.30, "Chinese": 0.45, "Fast Food": 0.40, "Other": 0.50}.get(cuisine, 0.50)
    veg_probs.append(p)
veg_flag = rng.random(N_ITEMS) < np.array(veg_probs)

# Popularity score: long-tail (power law) — most items rarely ordered
popularity_score = np.clip(rng.pareto(2.0, N_ITEMS), 0, 10)
popularity_score = (popularity_score / popularity_score.max()).round(4)

# Margin score: beverages & desserts have higher margins
margin_base = {"main": 0.25, "beverage": 0.55, "dessert": 0.50, "side": 0.35}
margin_score = np.array([
    np.clip(rng.normal(margin_base[c], 0.08), 0.05, 0.85)
    for c in item_categories
]).round(4)

menu_items_df = pd.DataFrame({
    "item_id":          np.arange(1, N_ITEMS + 1),
    "restaurant_id":    item_restaurant_ids,
    "category":         item_categories,
    "price":            item_prices,
    "veg_flag":         veg_flag,
    "popularity_score": popularity_score,
    "margin_score":     margin_score,
})

print(f"  Menu items shape: {menu_items_df.shape}")

# ──────────────────────────────────────────────
# 4. SESSIONS TABLE
# ──────────────────────────────────────────────
print("Generating sessions...")

# Generate timestamps over ~6 months
base_date = datetime(2024, 1, 1)

def sample_hour(n):
    """Simulate realistic hour distribution with lunch/dinner peaks."""
    # Weighted hour selection
    hour_weights = np.array([
        0.5,  # 0
        0.3,  # 1
        0.2,  # 2
        0.1,  # 3
        0.1,  # 4
        0.2,  # 5
        0.5,  # 6
        2.0,  # 7  breakfast start
        3.0,  # 8
        2.5,  # 9
        2.0,  # 10
        3.0,  # 11
        6.0,  # 12 lunch peak
        7.0,  # 13
        6.5,  # 14
        4.0,  # 15
        3.0,  # 16
        3.5,  # 17
        4.5,  # 18
        7.0,  # 19 dinner peak
        8.0,  # 20
        7.5,  # 21
        5.0,  # 22 late night
        3.0,  # 23
    ])
    hour_weights = hour_weights / hour_weights.sum()
    return rng.choice(np.arange(24), size=n, p=hour_weights)

session_user_ids = rng.integers(1, N_USERS + 1, N_SESSIONS)
session_hours    = sample_hour(N_SESSIONS)
day_offsets      = rng.integers(0, 180, N_SESSIONS)
timestamps       = [base_date + timedelta(days=int(d), hours=int(h),
                    minutes=int(rng.integers(0, 60)))
                    for d, h in zip(day_offsets, session_hours)]

# Meal time label
def hour_to_meal(h):
    if 7 <= h < 11:   return "breakfast"
    elif 11 <= h < 16: return "lunch"
    elif 16 <= h < 22: return "dinner"
    else:              return "late-night"

meal_times   = [hour_to_meal(h) for h in session_hours]
weekend_flag = np.array([ts.weekday() >= 5 for ts in timestamps])

# Match users to restaurants in the same city (realistic)
user_city_map = dict(zip(users_df["user_id"], users_df["city"]))
rest_city_map = dict(zip(restaurants_df["restaurant_id"], restaurants_df["city"]))

# Pre-build city → restaurant_id lookup
city_rest_lookup = restaurants_df.groupby("city")["restaurant_id"].apply(list).to_dict()

session_restaurant_ids = np.array([
    rng.choice(city_rest_lookup[user_city_map[uid]])
    for uid in session_user_ids
])

# Initial cart value: influenced by weekend + user segment
user_segment_map = dict(zip(users_df["user_id"], users_df["segment"]))
cart_value_base = np.array([
    {"budget": 120, "mid": 280, "premium": 550}[user_segment_map[uid]]
    for uid in session_user_ids
])
weekend_boost = np.where(weekend_flag, 1.15, 1.0)
cart_value_initial = (cart_value_base * weekend_boost * rng.uniform(0.7, 1.3, N_SESSIONS)).round(2)

sessions_df = pd.DataFrame({
    "session_id":        np.arange(1, N_SESSIONS + 1),
    "user_id":           session_user_ids,
    "restaurant_id":     session_restaurant_ids,
    "timestamp":         timestamps,
    "hour_of_day":       session_hours,
    "meal_time":         meal_times,
    "weekend_flag":      weekend_flag.astype(int),
    "cart_value_initial": cart_value_initial,
})

print(f"  Sessions shape: {sessions_df.shape}")

# ──────────────────────────────────────────────
# 5. CART EVENTS TABLE
# ──────────────────────────────────────────────
print("Generating cart events (~250K rows)... this may take ~30s")

# Pre-build restaurant → item lookup
rest_item_lookup = menu_items_df.groupby("restaurant_id")["item_id"].apply(list).to_dict()
item_cat_map      = dict(zip(menu_items_df["item_id"], menu_items_df["category"]))
item_price_map    = dict(zip(menu_items_df["item_id"], menu_items_df["price"]))
item_pop_map      = dict(zip(menu_items_df["item_id"], menu_items_df["popularity_score"]))
user_ps_map       = dict(zip(users_df["user_id"], users_df["price_sensitivity_score"]))
user_seg_map      = dict(zip(users_df["user_id"], users_df["segment"]))

# Target number of events per session (average ~2.5)
events_per_session = rng.integers(1, 6, N_SESSIONS)
# Scale to hit ~250K
scale_factor = TARGET_EVENTS / events_per_session.sum()
events_per_session = np.round(events_per_session * scale_factor).astype(int).clip(1, 8)

all_events = []
event_id   = 1

for idx, row in sessions_df.iterrows():
    sid   = row["session_id"]
    uid   = row["user_id"]
    rid   = row["restaurant_id"]
    hour  = row["hour_of_day"]
    wknd  = row["weekend_flag"]
    n_ev  = events_per_session[idx]

    # Available items for this restaurant
    avail_items = rest_item_lookup.get(rid, [])
    if not avail_items or n_ev == 0:
        continue

    # Sample items with popularity-weighted probability
    pops = np.array([item_pop_map[i] for i in avail_items]) + 0.01
    probs = pops / pops.sum()
    chosen_items = rng.choice(
        avail_items,
        size=min(n_ev, len(avail_items)),
        replace=False,
        p=probs
    )

    cart_value = row["cart_value_initial"]
    cart_cats  = set()
    ps         = user_ps_map.get(uid, 0.5)
    seg        = user_seg_map.get(uid, "mid")

    for add_order, item_id in enumerate(chosen_items, start=1):
        cat   = item_cat_map[item_id]
        price = item_price_map[item_id]
        pop   = item_pop_map[item_id]

        cart_value += price
        cart_cats.add(cat)

        # is_add_on_candidate: True for non-main items or items after first addition
        is_candidate = (cat in ["beverage", "dessert", "side"]) or (add_order > 1)

        # ── TARGET LABEL LOGIC ──
        # Realistic add-on acceptance: real-world rates are 5-20%.
        # Target overall acceptance rate: ~12-18%.
        if is_candidate:
            # 1. Cart completeness: incomplete meals have slightly higher
            #    acceptance (user is nudged to "complete" the meal)
            missing_bev  = "beverage" not in cart_cats
            missing_des  = "dessert"  not in cart_cats
            completeness_boost = 0.06 * missing_bev + 0.05 * missing_des

            # 2. Strong rejection bias for already-complete meals
            #    (cart already has both beverage AND dessert → low incentive)
            meal_complete_penalty = 0.05 if (not missing_bev and not missing_des) else 0.0

            # 3. User segment: premium users slightly more receptive,
            #    budget users slightly less receptive
            segment_boost = (
                0.025 if seg == "premium" else
                -0.015 if seg == "budget" else
                0.0
            )

            # 4. Dessert affinity for premium users ordering desserts
            dessert_boost = 0.05 if (seg == "premium" and cat == "dessert") else 0.0

            # 5. Item popularity: popular items convert slightly better
            pop_boost = pop * 0.12

            # 6. Price sensitivity penalty (strengthened)
            #    High price-sensitivity + expensive item → strong rejection
            price_penalty = ps * (price / 300) * 0.20

            # 7. Peak hour slight boost (social / group dining)
            peak_boost = 0.025 if (12 <= hour <= 15 or 19 <= hour <= 22) else 0.0

            # 8. Weekend small boost
            wknd_boost = 0.015 * wknd

            # Low base probability reflecting real-world add-on resistance
            p_accept = (0.10
                        + completeness_boost
                        + segment_boost
                        + dessert_boost
                        + pop_boost
                        + peak_boost
                        + wknd_boost
                        - price_penalty
                        - meal_complete_penalty)

            # Controlled noise: slightly negative mean to pull distribution
            # toward realistic low acceptance while adding real-world variance
            noise = rng.normal(-0.015, 0.04)
            p_accept = float(np.clip(p_accept + noise, 0.01, 0.40))

            add_on_accepted = int(rng.random() < p_accept)
        else:
            # First main item is always "accepted" (anchor item)
            add_on_accepted = 1
            p_accept = 1.0

        all_events.append({
            "event_id":           event_id,
            "session_id":         sid,
            "item_id":            item_id,
            "add_order":          add_order,
            "cart_value_after_add": round(cart_value, 2),
            "is_add_on_candidate": int(is_candidate),
            "add_on_accepted":    add_on_accepted,
        })
        event_id += 1

cart_events_df = pd.DataFrame(all_events)
print(f"  Cart events shape: {cart_events_df.shape}")

# ──────────────────────────────────────────────
# 6. FEATURE ENGINEERING → MODEL DATASET
# ──────────────────────────────────────────────
print("Engineering features for model dataset...")

# ── User features ──
user_order_totals = sessions_df.groupby("user_id")["cart_value_initial"].agg(["mean", "count"])
user_order_totals.columns = ["user_avg_spend", "session_count"]

# Frequency bucket
def freq_bucket(n):
    if n <= 3:   return 0   # cold-start
    elif n <= 15: return 1  # occasional
    elif n <= 50: return 2  # regular
    else:         return 3  # power user

users_df["user_frequency_bucket"] = users_df["lifetime_orders"].apply(freq_bucket)

# Dessert & beverage affinity: merge cart events with item categories
cart_with_cat = cart_events_df.merge(
    menu_items_df[["item_id", "category"]], on="item_id"
).merge(sessions_df[["session_id", "user_id"]], on="session_id")

def affinity(cat_name):
    cat_df = cart_with_cat[cart_with_cat["category"] == cat_name]
    accepted = cat_df.groupby("user_id")["add_on_accepted"].mean().rename(f"{cat_name}_affinity")
    return accepted

dessert_affinity  = affinity("dessert").rename("dessert_affinity_score")
beverage_affinity = affinity("beverage").rename("beverage_affinity_score")

users_features_df = users_df[["user_id", "segment", "avg_order_value",
                               "price_sensitivity_score", "recency_score",
                               "user_frequency_bucket"]].copy()
users_features_df = users_features_df.merge(user_order_totals, on="user_id", how="left")
users_features_df = users_features_df.merge(dessert_affinity,  on="user_id", how="left")
users_features_df = users_features_df.merge(beverage_affinity, on="user_id", how="left")
users_features_df[["dessert_affinity_score", "beverage_affinity_score"]] = \
    users_features_df[["dessert_affinity_score", "beverage_affinity_score"]].fillna(0.3)

# ── Restaurant features ──
restaurants_df["restaurant_popularity_rank"] = restaurants_df["order_volume"].rank(
    ascending=False, method="min").astype(int)

# ── Build model dataset from candidate add-on events ──
addon_events = cart_events_df[cart_events_df["is_add_on_candidate"] == 1].copy()

# Merge session context
addon_events = addon_events.merge(
    sessions_df[["session_id", "user_id", "restaurant_id",
                 "hour_of_day", "weekend_flag", "meal_time"]],
    on="session_id"
)

# Merge item features
addon_events = addon_events.merge(
    menu_items_df[["item_id", "category", "price",
                   "veg_flag", "popularity_score", "margin_score"]],
    on="item_id"
)

# Cart-level aggregates per session up to this add_order
# (for efficiency, compute from pre-grouped data)
session_cat_flags = cart_events_df.merge(
    menu_items_df[["item_id", "category"]], on="item_id"
).groupby("session_id").apply(
    lambda g: pd.Series({
        "cart_item_count":        len(g),
        "cart_has_beverage_flag": int((g["category"] == "beverage").any()),
        "cart_has_dessert_flag":  int((g["category"] == "dessert").any()),
        "cart_total_value":       g["cart_value_after_add"].max(),
    })
).reset_index()

addon_events = addon_events.merge(session_cat_flags, on="session_id", how="left")

# Missing meal component flag: no beverage OR no dessert in cart
addon_events["missing_meal_component_flag"] = (
    (addon_events["cart_has_beverage_flag"].fillna(0).astype(int) == 0) |
    (addon_events["cart_has_dessert_flag"].fillna(0).astype(int) == 0)
).astype(int)

# Peak hour flag
addon_events["peak_hour_flag"] = addon_events["hour_of_day"].apply(
    lambda h: 1 if (12 <= h <= 15 or 19 <= h <= 22) else 0
)

# City encoded (from user city via user_id)
city_enc = dict(zip(CITIES, range(len(CITIES))))
user_city_series = users_df.set_index("user_id")["city"]
addon_events["city_encoded"] = addon_events["user_id"].map(user_city_series).map(city_enc)

# Cuisine match: does user's preferred cuisine match restaurant cuisine?
user_pref_cuisine = users_df.set_index("user_id")["preferred_cuisine"]
rest_cuisine_series = restaurants_df.set_index("restaurant_id")["cuisine_type"]
addon_events["user_pref_cuisine"] = addon_events["user_id"].map(user_pref_cuisine)
addon_events["rest_cuisine"]      = addon_events["restaurant_id"].map(rest_cuisine_series)
addon_events["cuisine_match_score"] = (
    addon_events["user_pref_cuisine"] == addon_events["rest_cuisine"]
).astype(float)

# Restaurant popularity rank
rest_pop_rank = restaurants_df.set_index("restaurant_id")["restaurant_popularity_rank"]
addon_events["restaurant_popularity_rank"] = addon_events["restaurant_id"].map(rest_pop_rank)

# Merge user features
addon_events = addon_events.merge(
    users_features_df[["user_id", "user_avg_spend", "user_frequency_bucket",
                        "dessert_affinity_score", "beverage_affinity_score",
                        "price_sensitivity_score", "recency_score"]],
    on="user_id", how="left"
)

# ── Final model dataset columns ──
MODEL_COLS = [
    "event_id", "session_id", "user_id", "restaurant_id", "item_id",
    # User features
    "user_avg_spend", "user_frequency_bucket",
    "dessert_affinity_score", "beverage_affinity_score",
    "price_sensitivity_score", "recency_score",
    # Restaurant features
    "restaurant_popularity_rank", "cuisine_match_score",
    # Cart features
    "cart_item_count", "cart_has_beverage_flag", "cart_has_dessert_flag",
    "cart_total_value", "missing_meal_component_flag",
    # Item features
    "category", "price", "popularity_score", "margin_score",
    # Context features
    "peak_hour_flag", "weekend_flag", "city_encoded",
    "hour_of_day", "meal_time",
    # Target
    "add_on_accepted",
]

model_dataset_df = addon_events[MODEL_COLS].reset_index(drop=True)

print(f"  Model dataset shape: {model_dataset_df.shape}")
print(f"  Target balance: {model_dataset_df['add_on_accepted'].value_counts(normalize=True).round(3).to_dict()}")

# ──────────────────────────────────────────────
# SUMMARY & SAVE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"  users_df:         {users_df.shape}")
print(f"  restaurants_df:   {restaurants_df.shape}")
print(f"  menu_items_df:    {menu_items_df.shape}")
print(f"  sessions_df:      {sessions_df.shape}")
print(f"  cart_events_df:   {cart_events_df.shape}")
print(f"  model_dataset_df: {model_dataset_df.shape}")

# Save all tables as CSVs
import os
OUT_DIR = "csao_data"
os.makedirs(OUT_DIR, exist_ok=True)

for name, df in [("users", users_df), ("restaurants", restaurants_df),
                  ("menu_items", menu_items_df), ("sessions", sessions_df),
                  ("cart_events", cart_events_df), ("model_dataset", model_dataset_df)]:
    path = f"{OUT_DIR}/{name}.csv"
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        alt = f"{OUT_DIR}/{name}_new.csv"
        df.to_csv(alt, index=False)
        print(f"  ⚠ {path} locked — saved to {alt} instead")

print(f"\nAll CSVs saved to ./{OUT_DIR}/")

# ──────────────────────────────────────────────
# QUICK SANITY CHECKS
# ──────────────────────────────────────────────
print("\n── Sanity Checks ──")
print("User segments:\n", users_df["segment"].value_counts())
print("\nMeal time distribution:\n", sessions_df["meal_time"].value_counts())
print("\nItem category distribution:\n", menu_items_df["category"].value_counts())
print("\nAdd-on accepted rate by category:")
cat_rates = addon_events.groupby("category")["add_on_accepted"].mean().round(3)
print(cat_rates)
print("\nAdd-on accepted rate by user segment:")
seg_map = users_df.set_index("user_id")["segment"]
addon_events["segment"] = addon_events["user_id"].map(seg_map)
print(addon_events.groupby("segment")["add_on_accepted"].mean().round(3))
