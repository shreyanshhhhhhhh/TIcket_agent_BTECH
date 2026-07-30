# Routing Design

## Category to Department Mapping

| Category | Department |
|---|---|
| Infrastructure | Infrastructure & Systems Team |
| Application | Application Support Team |
| Security | Security Operations (SecOps) |
| Database | Database Administration (DBA) Team |
| Storage | Storage & Backup Team |
| Network | Network Operations Team |
| Access Management | Identity & Access Management (IAM) Team |

## Fallback
Any unrecognized category routes to "General IT Support" as a safety default.

## Design Rationale
This is a static lookup table for now, since the 7 categories map 1:1 to fixed departments in a typical IT services organization. If department structures were more dynamic (e.g., configurable per client/company), this would be moved to a database table instead of hardcoded Python — noted as a future extensibility point.