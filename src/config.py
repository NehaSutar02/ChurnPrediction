INDUSTRIES = {
    "telecom": {
        "path": "data/raw/telecom.csv",
        "target": "Churn",
        "id": "customerID",
    },
    "saas": {
        "path": "data/raw/saas.csv",
        "target": "Churn",
        "id": "customer_id",
        "synthetic": True,
    },
    "ott": {
        "path": "data/raw/ott.csv",
        "target": "Churn",
        "id": "customer_id",
        "synthetic": True,
    },
}
