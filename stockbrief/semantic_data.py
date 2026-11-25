"""Independent authored fixtures for pricing and input review."""
from .corpus import CALIBRATION, LABELS, TRAIN

SEMANTIC_LABELS = {**LABELS, "pricing": "Pricing & commercial updates", "review": "Needs review"}

SEMANTIC_TRAIN = {
    **TRAIN,
    "earnings": TRAIN["earnings"] + (
        "The retailer reported a half-year net loss after taking a one-off property write-down.",
        "Reporting a Q3 FY26 consolidated loss after an impairment at its European subsidiary.",
        "The group posted a first-quarter loss as factory closure expenses weighed on its results.",
        "The electronics maker swung to a loss in the September quarter after recording an exceptional tax charge.",
        "While reporting a full-year operating loss, management disclosed exceptional costs in its overseas unit.",
        "The company reported a Q4 FY25 net loss because of severance charges and lower demand.",
        "One-off expenses at the group's European rail division drove a quarterly loss after tax.",
        "A goodwill impairment widened the consolidated net loss in the latest reporting period.",
    ),
    "pricing": (
        "The carmaker will increase ex-showroom prices across its portfolio because component costs have risen.",
        "The automaker announced a list price rise for selected diesel and hybrid vehicles.",
        "The group will reduce retail prices for several models starting next month.",
        "An increase in raw material and freight costs will lead to a new price list for its appliances.",
        "The company has revised its recommended retail price for its entry-level phone.",
        "Dealerships are offering festival discounts on selected vehicle variants.",
        "The manufacturer launched a new electric model at an introductory ex-showroom price.",
        "The telecom operator increased tariffs across its prepaid plans.",
        "The retailer announced cashback and trade-in discounts on eligible purchases.",
        "The airline revised ticket fares on regional services following a rise in fuel costs.",
        "The company raised the retail price of its flagship SUV by two percent.",
        "The brand introduced a limited-time exchange bonus on its premium hatchbacks.",
        "Buyers will pay less for selected models after the company cut showroom prices.",
        "The new consumer price list increases the cost of each unit from the next billing cycle.",
    ),
    "review": (
        "Please acknowledge receipt of this message and circulate it to the relevant team.",
        "The document does not give specific details for the item referenced in the covering letter.",
        "We will issue a response when we have information to share.",
        "This note forwards material from an earlier correspondence for reference.",
        "The statement offers no specifics about the reports under discussion.",
        "An informational update will be published at a future time.",
        "See the accompanying document for context about this announcement.",
        "The issuer gave no additional details about the issue.",
        "The letter contains a general acknowledgement and an expression of thanks.",
        "Further information about this topic is still awaited.",
        "Mix the flour and butter before baking the mixture until golden brown.",
        "My laptop keeps turning on when I close the lid. How can I fix it?",
        "Write a short poem about walking home in the rain.",
        "The doctor advised resting the ankle for two weeks.",
        "The home team won the match after a late goal in stoppage time.",
        "The software tutorial explains installing a package and opening a command prompt.",
    ),
}

SEMANTIC_CALIBRATION = {
    **CALIBRATION,
    "pricing": (
        "The manufacturer is increasing showroom charges for selected passenger cars next month.",
        "Customers will receive a holiday rebate when they exchange their used vehicle for a new model.",
        "The retailer lowered the shelf price of its own-brand tablets after reducing shipping costs.",
        "The company unveiled a hatchback with a starting price of ₹9 lakh.",
    ),
    "review": CALIBRATION["review"] + (
        "The trail around the lake is open to hikers again after yesterday's rain.",
        "Use a spreadsheet to plan tomorrow's grocery shopping and meal preparation.",
        "This newsletter collects several links and invites readers to write back.",
        "The document provides little detail about what was decided or when it takes effect.",
    ),
}

SEMANTIC_ADDITIONAL_TEST = {
    "pricing": (
        "The manufacturer will pass rising battery costs to buyers by lifting showroom prices next week.",
        "The company cut the consumer tariff for its annual broadband plan.",
        "Selected crossover variants are eligible for a cashback of ₹40,000 through the end of the month.",
        "The new motorcycle was unveiled at an introductory price of ₹1.8 lakh.",
    ),
    "review": (
        "What is the easiest way to cook plain rice in a saucepan?",
        "I need to move our camping trip because the campsite will be closed.",
        "The author describes how to paint a mural with watercolours.",
        "Further details of the subject are contained in the enclosed correspondence.",
    ),
}
