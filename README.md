# FastAPI AI Orchestration Service

This service acts as the orchestration layer between your frontend applications, Google's Gemini LLM, and your .NET 9 backend running on Railway.

## Architecture Request Flow

1. **User Input:** Frontend sends a chat message to this service (`/api/chat`).
2. **Intent mapping:** This service queries Gemini 2.5 to determine the user's intent and extracts relevant parameters.
3. **Tool routing:** It checks the `ToolRegistry` for a matching .NET backend endpoint.
4. **Execution:** If a tool is found, this service securely forwards the request to the .NET backend.
5. **Response:** A unified, structured JSON response is returned to the frontend.

## Local Setup

1. **Install Dependencies:**
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in your keys.
   ```bash
   cp .env.example .env
   ```
   *Note: Ensure `DOTNET_BACKEND_URL` does NOT end with a trailing slash.*

3. **Run the Server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The API will be available at `http://localhost:8000`.
   Swagger UI is available at `http://localhost:8000/docs`.

## Railway Deployment

This project is built to deploy seamlessly on Railway.

1. Create a new service in your Railway project from your GitHub repository.
2. Railway will automatically detect the Python environment and `requirements.txt`.
3. Go to the **Variables** tab in Railway and add:
   - `GEMINI_API_KEY` = your Gemini key
   - `DOTNET_BACKEND_URL` = exactly as shown in your .NET Railway deployment (e.g., `https://my-backend.up.railway.app`)
   - `ENVIRONMENT` = `production`
   - `PORT` = `8000` (Railway will export its own PORT context)

To ensure Railway starts the app correctly, it is recommended to add this line as a custom Start Command in Railway settings:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Testing / Sample Request

### Health Check (GET)
```bash
curl http://localhost:8000/health
```

### Chat Request (POST)
```bash
curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Schedule a meeting for tomorrow at 2 PM", "user_id": "123"}'
```

**Expected JSON Response:**
```json
{
  "response": "Successfully executed backend tool for intent 'ScheduleMeeting'.",
  "intent_executed": "ScheduleMeeting",
  "backend_data": {
    "status": "success",
    "meeting_id": 42
  }
}
```

## Adding New Tools

To add a new tool in the future:
1. Ensure the .NET execution layer is handling the specific route.
2. Add the intent prompt to the `system_instruction` in `app/services/ai_service.py`.
3. Map the intent to the specific .NET route in `app/services/tool_registry.py`.
