from omegaconf import OmegaConf
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

db_cfg = OmegaConf.load("configs/_base_/db.yaml")

username, password, host, port, database = (
    db_cfg.username,
    db_cfg.password,
    db_cfg.host,
    db_cfg.port,
    db_cfg.database,
)
SQLALCHEMY_DATABASE_URL = f"mysql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Perflog(Base):
    __tablename__ = "perflog"
    id = Column(Integer, primary_key=True, index=True)
    task = Column(String)
    model_type = Column(String)
    model_name = Column(String)
    performance = Column(String)
    config = Column(String)
    record_time = Column(Integer)


def process_and_upload_performance():
    db = SessionLocal()
    perflogs = db.query(Perflog).all()
    for perflog in perflogs:
        print(perflog.task)
        print(perflog.model_type)
        print(perflog.model_name)
        print(perflog.performance)
        print(perflog.config)
        print(perflog.record_time)
        print("==========================")
    db.close()
    return perflogs


if __name__ == "__main__":
    process_and_upload_performance()
