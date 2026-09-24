"""ASGI entrypoint: `uvicorn rootcause.api.main:app`.

Loads .env here (not in app.py) so importing create_app() in tests never
picks up a real OPENAI_API_KEY.
"""

from dotenv import load_dotenv

load_dotenv()

from rootcause.api.app import create_app  # noqa: E402

app = create_app()
