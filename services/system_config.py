from sqlalchemy.orm import Session

from models import SystemConfigDB


def get_system_config(db: Session, key: str, default: str = "") -> str:
    row = db.query(SystemConfigDB).filter(SystemConfigDB.key == key).first()
    return row.value if row else default


def set_system_config(db: Session, key: str, value: str) -> None:
    row = db.query(SystemConfigDB).filter(SystemConfigDB.key == key).first()
    if row:
        row.value = value
    else:
        db.add(SystemConfigDB(key=key, value=value))
    db.commit()
