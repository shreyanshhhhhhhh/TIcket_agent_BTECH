from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'ticket_system.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    role = Column(String, nullable=False)  # "employee" | "agent"
    department = Column(String, nullable=True)  # only for agents
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    tickets = relationship("Ticket", back_populates="employee")


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    classifier_confidence = Column(Float, nullable=True)
    priority = Column(String, nullable=True)
    department = Column(String, nullable=True)
    status = Column(String, default="open")
    rag_suggested_resolution = Column(Text, nullable=True)
    best_similarity_score = Column(Float, nullable=True)
    employee_feedback = Column(String, nullable=True)  # "yes" | "partially_yes" | "no"
    employee_followup_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("User", back_populates="tickets")
    resolution = relationship("Resolution", back_populates="ticket", uselist=False)


class Resolution(Base):
    __tablename__ = "resolutions"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"))
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    root_cause = Column(Text, nullable=True)
    fix_steps = Column(Text, nullable=True)
    resolution_status = Column(String, default="new")  # validated | improved | new | failed_suggestion
    parent_resolution_id = Column(Integer, ForeignKey("resolutions.id"), nullable=True)
    success_count = Column(Integer, default=0)
    partial_count = Column(Integer, default=0)
    fail_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="resolution")


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()