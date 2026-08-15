from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from database import get_db, init_db, User, Ticket, Resolution
from agent import process_ticket

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="AI Ticket Routing & Resolution System")

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/app", include_in_schema=False)
def serve_home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/app/employee", include_in_schema=False)
def serve_employee():
    return FileResponse(os.path.join(FRONTEND_DIR, "employee.html"))

@app.get("/app/agent", include_in_schema=False)
def serve_agent():
    return FileResponse(os.path.join(FRONTEND_DIR, "agent.html"))

@app.get("/app/admin", include_in_schema=False)
def serve_admin():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))




class TicketCreate(BaseModel):
    employee_id: int
    title: str
    description: str


class FeedbackInput(BaseModel):
    feedback: str
    followup_notes: Optional[str] = None


class ResolutionInput(BaseModel):
    resolved_by: int
    root_cause: str
    fix_steps: str


@app.post("/submit-ticket")
def submit_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    full_text = f"{payload.title}. {payload.description}" if payload.title else payload.description
    result = process_ticket(full_text)

    status = "auto_resolved" if result["decision"] == "auto_resolve" else "escalated_full"

    ticket = Ticket(
        employee_id=payload.employee_id,
        title=payload.title,
        description=payload.description,
        category=result["category"],
        classifier_confidence=result["classifier_confidence"],
        priority="Medium",
        department=result["department"],
        status=status,
        rag_suggested_resolution=result["suggested_resolution"],
        best_similarity_score=result["best_similarity"],
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    return {
        "ticket_id": ticket.id,
        "category": ticket.category,
        "department": ticket.department,
        "decision": result["decision"],
        "suggested_resolution": ticket.rag_suggested_resolution,
        "reason": result["reason"],
    }


@app.post("/ticket-feedback/{ticket_id}")
def submit_feedback(ticket_id: int, payload: FeedbackInput, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket.employee_feedback = payload.feedback
    ticket.employee_followup_notes = payload.followup_notes

    if payload.feedback == "yes":
        ticket.status = "resolved"
    elif payload.feedback == "partially_yes":
        ticket.status = "escalated_partial"
    else:
        ticket.status = "escalated_full"

    db.commit()
    return {"message": "Feedback recorded", "new_status": ticket.status}


@app.get("/tickets/assigned/{department}")
def get_assigned_tickets(department: str, db: Session = Depends(get_db)):
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

    tickets = db.query(Ticket).filter(
        Ticket.department == department,
        Ticket.status.in_(["escalated_full", "escalated_partial"])
    ).all()

    tickets_sorted = sorted(tickets, key=lambda t: priority_order.get(t.priority, 4))

    return [
        {
            "id": t.id,
            "title": t.title,
            "description": t.description,
            "priority": t.priority,
            "status": t.status,
            "rag_suggested_resolution": t.rag_suggested_resolution,
            "employee_followup_notes": t.employee_followup_notes,
        }
        for t in tickets_sorted
    ]


@app.post("/resolve-ticket/{ticket_id}")
def resolve_ticket(ticket_id: int, payload: ResolutionInput, db: Session = Depends(get_db)):
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    resolution = Resolution(
        ticket_id=ticket_id,
        resolved_by=payload.resolved_by,
        root_cause=payload.root_cause,
        fix_steps=payload.fix_steps,
        resolution_status="new",
    )
    db.add(resolution)

    ticket.status = "resolved"
    db.commit()

    return {"message": "Resolution recorded", "ticket_id": ticket_id}


@app.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    total = db.query(Ticket).count()
    auto_resolved = db.query(Ticket).filter(Ticket.status == "auto_resolved").count()
    escalated = db.query(Ticket).filter(Ticket.status.in_(["escalated_full", "escalated_partial"])).count()
    resolved = db.query(Ticket).filter(Ticket.status == "resolved").count()

    return {
        "total_tickets": total,
        "auto_resolved": auto_resolved,
        "escalated": escalated,
        "resolved": resolved,
        "auto_resolve_rate": round(auto_resolved / total, 2) if total else 0,
    }


@app.get("/")
def root():
    return {"message": "AI Ticket Routing & Resolution API is running"}