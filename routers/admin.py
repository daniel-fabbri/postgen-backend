from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_db
from dependencies import get_current_user, _require_admin
from models import UserDB
from schemas import UserOut
from services.system_config import get_system_config, set_system_config

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users", response_model=list[UserOut])
def list_all_users(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    users = db.query(UserDB).order_by(UserDB.created_at.desc()).all()
    return [UserOut.model_validate(u) for u in users]


@router.get("/rates")
def get_rates(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    credits_per_real = float(get_system_config(db, "credits_per_real", "1.0"))
    initial_credits = float(get_system_config(db, "initial_credits", "0.0"))
    return {"credits_per_real": credits_per_real, "initial_credits": initial_credits}


@router.put("/rates")
def update_rates(
    payload: dict,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    credits_per_real = float(payload.get("credits_per_real", 1.0))
    initial_credits = float(payload.get("initial_credits", 0.0))
    if credits_per_real <= 0:
        raise HTTPException(status_code=422, detail="credits_per_real deve ser positivo")
    if initial_credits < 0:
        raise HTTPException(status_code=422, detail="initial_credits não pode ser negativo")
    set_system_config(db, "credits_per_real", str(credits_per_real))
    set_system_config(db, "initial_credits", str(initial_credits))
    return {"credits_per_real": credits_per_real, "initial_credits": initial_credits}


@router.post("/users/{user_id}/reset-credits")
def admin_reset_credits(
    user_id: int,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    user.credits_balance = 0.0
    db.execute(text("DELETE FROM credit_usage WHERE user_id = :uid"), {"uid": user_id})
    db.commit()
    return {"ok": True, "user_id": user_id}
