from sqlmodel import Session, create_engine

from app.config.settings import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)


def get_session():
    with Session(engine) as session:
        yield session
