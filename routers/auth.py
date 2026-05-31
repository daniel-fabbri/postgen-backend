from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from database import get_db
from dependencies import hash_password, verify_password, create_access_token, get_current_user
from models import UserDB
from schemas import UserRegister, UserLogin, UserOut, TokenOut, UserUpdate
from services.system_config import get_system_config

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=201)
def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == data.email).first():
        raise HTTPException(status_code=409, detail="E-mail já cadastrado")
    initial_credits = float(get_system_config(db, "initial_credits", "0.0"))
    user = UserDB(
        email=data.email,
        password_hash=hash_password(data.password),
        name=data.name,
        credits_balance=initial_credits,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.email)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut)
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="E-mail ou senha incorretos")
    token = create_access_token(user.email)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(current_user: UserDB = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


# ---------------------------------------------------------------------------
# User profile (logicamente ligado a auth)
# ---------------------------------------------------------------------------
users_router = APIRouter(prefix="/api/users", tags=["users"])


@users_router.put("/profile", response_model=UserOut)
def update_profile(
    data: UserUpdate,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if data.email != current_user.email:
        existing = db.query(UserDB).filter(
            UserDB.email == data.email,
            UserDB.id != current_user.id,
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="E-mail já está em uso")
    current_user.name = data.name
    current_user.email = data.email
    db.commit()
    db.refresh(current_user)
    return UserOut.model_validate(current_user)
