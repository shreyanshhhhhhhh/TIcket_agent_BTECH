I Finalized 3-outcome design

1. YES (fully resolved)

Ticket marked as closed
The RAG-suggested resolution is confirmed as validated and saved back into the knowledge base
If the same/similar issue occurs again, this resolution now has proven track record — RAG can retrieve it with higher confidence next time
No human agent involved at all — full self-service success

2. PARTIALLY YES (partially resolved)

Ticket is escalated to the correction/support team, but NOT as a blank/fresh ticket
The system passes along:
The original problem description
The RAG-suggested resolution that was tried
What the employee said happened (e.g., "this fixed the login issue but VPN still disconnects")
The agent starts from this partial progress instead of starting over
Once agent fixes the remaining issue, their new fix is saved as an update/improvement to that resolution (not a brand-new unrelated one) — this is important, it means your knowledge base is refining existing solutions, not just accumulating disconnected ones

3. NO (not resolved / escalated)

Ticket is escalated to the correction team as a genuine miss
The system flags: "RAG-suggested resolution did not work for this case"
Agent resolves it fresh
Their fix gets saved as a new resolution in the knowledge base
Optionally: the original RAG suggestion that failed gets a small negative signal (so it doesn't keep getting retrieved for this type of issue as confidently)
Why this design is solid (for your documentation)
YES → reinforces good solutions, makes RAG stronger over time
PARTIALLY YES → prevents duplicate work for the agent, and improves existing solutions rather than fragmenting your knowledge base with near-duplicate entries
NO → correctly identifies knowledge gaps, feeds fresh, verified fixes back in, and slightly demotes resolutions that don't actually work

This is a genuinely well-thought-out feedback loop — it's the actual mechanism that makes your "self-improving system" claim true and demonstrable, not just a marketing line in your report.