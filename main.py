import os
import requests
import chainlit as cl
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.tool import function_tool
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

load_dotenv()

# Initialize Weather Api Key
weather_api_key = os.getenv("WEATHER_API_KEY")

# Initialize Gemini API client
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=provider)

config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)

# Tool 1: Explain Weather Concepts

class WeatherPatternInput(BaseModel):
    question: str

@function_tool("weather_patterns")
def weather_patterns(input: WeatherPatternInput) -> str:
    question = input.question.lower()

    if "climate change" in question and "extreme weather" in question:
        return (
            " Climate change is closely linked to extreme weather events.\n\n"
            "As global temperatures rise due to increased greenhouse gas emissions:\n"
            "- Heatwaves become more frequent and intense\n"
            "- Warmer air holds more moisture, increasing heavy rainfall and flooding\n"
            "- Changing ocean temperatures fuel stronger hurricanes\n"
            "- Shifting weather patterns cause droughts in some areas and floods in others\n\n"
            "These changes are backed by scientific studies and observed trends over recent decades."
        )

    return (
        "Weather patterns are influenced by air pressure, temperature, humidity, wind, and solar radiation.\n\n"
        "For example:\n"
        "- Jet streams influence weather across continents\n"
        "- Mountains can create rain shadows (wet on one side, dry on the other)\n"
        "- Seasonal patterns cause monsoons and dry seasons in some regions\n\n"
        "Feel free to ask me more about a specific weather phenomenon!"
    )

# Tool 2: Live Weather via WeatherApi

@dataclass
class WeatherInfo:
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int
    location_name: str
    rain_1h: Optional[float] = None
    visibility: Optional[int] = None

@function_tool("get_weather")
def get_weather(location: str, unit: str = "C") -> str:
    WEATHER_API_KEY = weather_api_key
    if not WEATHER_API_KEY:
        return "❌ Weather API key is missing."

    units = "metric" if unit.upper() == "C" else "imperial"

    url = "https://api.weatherapi.com/v1/current.json"
    params = {
        "key": WEATHER_API_KEY,   
        "q": location             
}

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"❌ Failed to get weather data: {response.status_code} - {response.json().get('message', 'Unknown error')}"

        data = response.json()

        weather_info = WeatherInfo(
            temperature=data["current"]["temp_c"],
            feels_like=data["current"]["feelslike_c"],
            humidity=data["current"]["humidity"],
            description=data["current"]["condition"]["text"],
            wind_speed=data["current"]["wind_kph"] / 3.6,  
            pressure=data["current"]["pressure_mb"],
            location_name=data["location"]["name"]
        )


        report = f"""📍 **Weather in {weather_info.location_name}**
            🌡️ Temperature: {weather_info.temperature}°{unit.upper()} (Feels like {weather_info.feels_like}°{unit.upper()})
            ☁️ Conditions: {weather_info.description}
            💧 Humidity: {weather_info.humidity}%
            🌬️ Wind Speed: {weather_info.wind_speed} m/s
            🔽 Pressure: {weather_info.pressure} hPa
            """
        if weather_info.rain_1h:
            report += f"🌧️ Rain (last 1h): {weather_info.rain_1h} mm\n"
        if weather_info.visibility:
            report += f"👁️ Visibility: {weather_info.visibility / 1000:.1f} km\n"

        return report

    except Exception as e:
        return f"⚠️ An unexpected error occurred: {str(e)}"

    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching weather data: {str(e)}"

# Step 4: Agent Configuration

agent = Agent(
    name="Wiz WEATHER",
    instructions="""
You are a weather expert assistant named Wiz WEATHER.

You MUST:
- Only answer questions that are weather-related.
- Politely decline and redirect if the question is unrelated to weather, climate, or the environment.
- Describe seasonal and regional weather systems
- Discuss how climate change relates to weather phenomena like droughts, storms, and heatwaves
- Use tools to fetch or explain weather topics
""",
    tools=[weather_patterns, get_weather],
    model=model,
)

# Chainlit Event Handlers

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 **Welcome to _Wiz WEATHER!_**\n\nI'm your climate-savvy guide 🌦️.\nAsk me about current weather or anything weather-related!"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(agent, input=history, run_config=config)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)