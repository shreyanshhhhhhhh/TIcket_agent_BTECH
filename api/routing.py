"""
Routing logic: maps a predicted ticket category to the responsible support team/department.
"""

CATEGORY_TO_DEPARTMENT = {
    "Infrastructure": "Infrastructure & Systems Team",
    "Application": "Application Support Team",
    "Security": "Security Operations (SecOps)",
    "Database": "Database Administration (DBA) Team",
    "Storage": "Storage & Backup Team",
    "Network": "Network Operations Team",
    "Access Management": "Identity & Access Management (IAM) Team",
}

def route_ticket(category: str) -> str:
    """
    Given a predicted category, return the department responsible for handling it.
    Falls back to a general IT team if category is unrecognized.
    """
    return CATEGORY_TO_DEPARTMENT.get(category, "General IT Support")


if __name__ == "__main__":
    # Quick test
    test_categories = ["Network", "Security", "Unknown Category"]
    for cat in test_categories:
        print(f"{cat} -> {route_ticket(cat)}")