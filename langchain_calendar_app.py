import os
from datetime import datetime, timedelta
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI  # or ChatOpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    filters,
)
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import smtplib
from email.message import EmailMessage
from sqlalchemy import create_engine, text

# --- Load Environment ---
load_dotenv()

# --- Constants ---
(
    PARSE_INITIAL_INPUT,
    EDIT_ACTION,
    EDIT_PERSON,
    EDIT_TIME,
    EDIT_TOPIC,
    CONFIRM_DETAILS,
) = range(6)

# --- Database & API Clients ---
engine = create_engine(os.getenv("POSTGRES_URL"))
calendar = build(
    "calendar",
    "v3",
    credentials=Credentials.from_authorized_user_file(
        os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
    ),
)

# --- LangChain Setup ---
class CalendarAction(BaseModel):
    action: str = Field(description="create, update, or delete")
    person: str = Field(description="attendee name")
    time: str = Field(description="meeting time in ISO format")
    topic: str = Field(description="meeting purpose", default=None)

prompt = ChatPromptTemplate.from_template("Extract fields from: {input}")
model = ChatVertexAI(model="gemini-pro")  # or ChatOpenAI(model="gpt-4")
chain = prompt | model.with_structured_output(CalendarAction)

# --- State Management ---
user_sessions: Dict[int, Dict[str, Any]] = {}

# --- Core Functions ---
def get_email(name: str) -> str:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT email FROM contacts WHERE name = :name"), {"name": name})
        return result.scalar()

def is_slot_free(time: str) -> bool:
    events = (
        calendar.freebusy()
        .query(
            body={
                "timeMin": time,
                "timeMax": (datetime.fromisoformat(time) + timedelta(hours=1)).isoformat(),
                "items": [{"id": "primary"}],
            }
        )
        .execute()
    )
    return not events["calendars"]["primary"]["busy"]

def suggest_slots(original_time: str) -> list[str]:
    original_dt = datetime.fromisoformat(original_time)
    slots = []
    for delta in [-1, 1, -2, 2]:  # Check ±1h and ±2h
        new_time = (original_dt + timedelta(hours=delta)).isoformat()
        if is_slot_free(new_time):
            slots.append(new_time)
            if len(slots) == 2:
                break
    return slots

def send_email(to: str, content: str):
    msg = EmailMessage()
    msg.set_content(content)
    msg["Subject"] = "Meeting Update"
    msg["From"] = os.getenv("SMTP_USER")
    msg["To"] = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
        server.send_message(msg)

# --- Telegram Handlers ---
async def start_booking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_input = update.message.text
    try:
        extracted_data = chain.invoke({"input": user_input})
        chat_id = update.message.chat_id
        user_sessions[chat_id] = {
            "action": extracted_data.action,
            "person": extracted_data.person,
            "time": extracted_data.time,
            "topic": extracted_data.topic,
        }
        await show_editable_summary(update, chat_id)
        return CONFIRM_DETAILS
    except Exception as e:
        await update.message.reply_text(f"❌ Error parsing input: {e}")
        return ConversationHandler.END

async def show_editable_summary(update: Update, chat_id: int):
    data = user_sessions[chat_id]
    keyboard = [
        [InlineKeyboardButton("✏️ Action", callback_data="edit_action")],
        [InlineKeyboardButton("✏️ Person", callback_data="edit_person")],
        [InlineKeyboardButton("✏️ Time", callback_data="edit_time")],
        [InlineKeyboardButton("✏️ Topic", callback_data="edit_topic")],
        [InlineKeyboardButton("✅ Confirm", callback_data="confirm")],
    ]
    text = (
        f"*Current Details:*\n"
        f"Action: {data['action']}\n"
        f"Person: {data['person']}\n"
        f"Time: {data['time']}\n"
        f"Topic: {data['topic'] or 'None'}"
    )
    if isinstance(update, Update):
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    else:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

# --- Edit Handlers ---
async def edit_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("Send new action (create/update/delete):")
    return EDIT_ACTION

async def save_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.message.chat_id
    user_sessions[chat_id]["action"] = update.message.text
    await show_editable_summary(update, chat_id)
    return CONFIRM_DETAILS

# ... (similar handlers for person/time/topic)

# --- Confirmation Handler ---
async def confirm_details(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data = user_sessions[chat_id]
    
    if not is_slot_free(data["time"]):
        alternatives = suggest_slots(data["time"])
        keyboard = [
            [InlineKeyboardButton(f"Use {datetime.fromisoformat(slot).strftime('%H:%M')}", callback_data=f"use_alt_{slot}")]
            for slot in alternatives
        ]
        await query.edit_message_text(
            "⚠️ Slot unavailable. Choose alternative:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        return CONFIRM_DETAILS
    
    # Execute booking
    attendee_email = get_email(data["person"])
    if attendee_email:
        send_email(attendee_email, f"Meeting {data['action']}ed at {data['time']}")
    await query.edit_message_text("✅ Booking confirmed!")
    del user_sessions[chat_id]  # Cleanup
    return ConversationHandler.END

async def use_alternative_slot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    slot = query.data.replace("use_alt_", "")
    chat_id = query.message.chat_id
    user_sessions[chat_id]["time"] = slot
    await show_editable_summary(update, chat_id)
    return CONFIRM_DETAILS

# --- Main ---
def main():
    app = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND, start_booking)],
        states={
            EDIT_ACTION: [MessageHandler(filters.TEXT, save_action)],
            EDIT_PERSON: [MessageHandler(filters.TEXT, save_person)],
            EDIT_TIME: [MessageHandler(filters.TEXT, save_time)],
            EDIT_TOPIC: [MessageHandler(filters.TEXT, save_topic)],
            CONFIRM_DETAILS: [
                CallbackQueryHandler(edit_action, pattern="^edit_action$"),
                CallbackQueryHandler(edit_person, pattern="^edit_person$"),
                CallbackQueryHandler(edit_time, pattern="^edit_time$"),
                CallbackQueryHandler(edit_topic, pattern="^edit_topic$"),
                CallbackQueryHandler(confirm_details, pattern="^confirm$"),
                CallbackQueryHandler(use_alternative_slot, pattern="^use_alt_"),
            ],
        },
        fallbacks=[],
    )
    
    app.add_handler(conv_handler)
    app.run_polling()

if __name__ == "__main__":
    main()