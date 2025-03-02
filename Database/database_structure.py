from sqlalchemy import create_engine, ForeignKey, String, Integer, Column, Date, Float, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

Base = declarative_base()

class WeatherPrediction(Base):
    __tablename__ = "WeatherPrediction"
    RedcordID = Column(Integer, primary_key=True, autoincrement=True)
    Location = Column(String)
    Date = Column(Date)
    Temperature_C = Column(Float)
    Humidity_pct = Column(Float)
    Precipitation_mm = Column(Float)
    Wind_Speed_kmh = Column(Float)
    will_rain = Column(Boolean)

    def __init__(self, RedcordID, Location, Date, Temperature_C, Humidity_pct, Wind_Speed_kmh):
        self.RedcordID = RedcordID
        self.Location = Location
        self.Date = Date
        self.Temperature_C = Temperature_C
        self.Humidity_pct = Humidity_pct
        self.Precipitation_mm = Precipitation_mm
        self.Wind_Speed_kmh = Wind_Speed_kmh


DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()