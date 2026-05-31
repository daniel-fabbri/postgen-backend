from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

import mercadopago

from config import BASE_URL, MERCADOPAGO_ACCESS_TOKEN
from database import get_db
from dependencies import get_current_user
from models import UserDB, PaymentDB
from schemas import PaymentCreate, PaymentOut
from services.system_config import get_system_config

router = APIRouter(prefix="/api/payments", tags=["payments"])


@router.get("/rates")
def get_public_rates(db: Session = Depends(get_db)):
    credits_per_real = float(get_system_config(db, "credits_per_real", "1.0"))
    return {"credits_per_real": credits_per_real}


@router.post("/create")
def create_payment(
    payment_data: PaymentCreate,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not MERCADOPAGO_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Mercado Pago não configurado")
    try:
        sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)
        credits_per_real = float(get_system_config(db, "credits_per_real", "1.0"))
        credits_amount = round(payment_data.amount * credits_per_real, 2)
        payment_request = {
            "transaction_amount": float(payment_data.amount),
            "description": f"Compra de {credits_amount} créditos PostGen",
            "payment_method_id": "pix",
            "payer": {
                "email": current_user.email,
                "first_name": current_user.name.split()[0] if current_user.name else "Cliente",
            },
            "notification_url": f"{BASE_URL}/api/payments/webhook",
        }
        payment_response = sdk.payment().create(payment_request)
        payment = payment_response["response"]
        if payment_response["status"] not in [200, 201]:
            mp_error = payment.get("message") or payment.get("error") or str(payment)
            raise ValueError(f"Mercado Pago error {payment_response['status']}: {mp_error}")
        mp_payment_id = str(payment["id"])
        qr_code_base64 = payment.get("point_of_interaction", {}).get("transaction_data", {}).get("qr_code_base64", "")
        qr_code_data = payment.get("point_of_interaction", {}).get("transaction_data", {}).get("qr_code", "")
        db_payment = PaymentDB(
            user_id=current_user.id,
            mp_payment_id=mp_payment_id,
            amount=payment_data.amount,
            credits_amount=credits_amount,
            status="pending",
            qr_code=qr_code_base64,
            qr_code_data=qr_code_data,
        )
        db.add(db_payment)
        db.commit()
        db.refresh(db_payment)
        return {
            "payment_id": db_payment.id,
            "mp_payment_id": mp_payment_id,
            "amount": payment_data.amount,
            "credits_amount": credits_amount,
            "status": "pending",
            "qr_code": qr_code_base64,
            "qr_code_data": qr_code_data,
        }
    except Exception as e:
        print(f"[MP] create_payment exception: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Erro ao criar pagamento: {str(e)}")


@router.post("/webhook")
async def payment_webhook(
    request: dict,
    db: Session = Depends(get_db),
):
    try:
        if request.get("type") != "payment":
            return {"status": "ignored"}
        mp_payment_id = str(request.get("data", {}).get("id", ""))
        if not mp_payment_id:
            return {"status": "error", "message": "Payment ID not found"}
        payment = db.query(PaymentDB).filter(PaymentDB.mp_payment_id == mp_payment_id).first()
        if not payment:
            return {"status": "error", "message": "Payment not found in database"}
        sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)
        payment_info = sdk.payment().get(mp_payment_id)
        if payment_info["status"] != 200:
            return {"status": "error", "message": "Failed to get payment info"}
        mp_status = payment_info["response"].get("status", "")
        old_status = payment.status
        user = db.query(UserDB).filter(UserDB.id == payment.user_id).first()
        if user:
            if mp_status == "approved" and old_status != "approved":
                user.credits_balance += payment.credits_amount
            elif old_status == "approved" and mp_status in ("cancelled", "refunded", "charged_back"):
                user.credits_balance = max(0.0, user.credits_balance - payment.credits_amount)
        payment.status = mp_status
        db.commit()
        return {"status": "success", "payment_status": mp_status}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/my")
def list_my_payments(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    payments = (
        db.query(PaymentDB)
        .filter(PaymentDB.user_id == current_user.id)
        .order_by(PaymentDB.created_at.desc())
        .all()
    )
    return [
        {
            "id": p.id,
            "amount": p.amount,
            "credits_amount": p.credits_amount,
            "status": p.status,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in payments
    ]


@router.get("/{payment_id}")
def get_payment_status(
    payment_id: int,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    payment = db.query(PaymentDB).filter(
        PaymentDB.id == payment_id,
        PaymentDB.user_id == current_user.id,
    ).first()
    if not payment:
        raise HTTPException(status_code=404, detail="Pagamento não encontrado")
    try:
        sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)
        payment_info = sdk.payment().get(payment.mp_payment_id)
        if payment_info["status"] == 200:
            mp_status = payment_info["response"].get("status", payment.status)
            if mp_status != payment.status:
                old_status = payment.status
                payment.status = mp_status
                payment.updated_at = func.now()
                if mp_status == "approved" and old_status != "approved":
                    current_user.credits_balance += payment.credits_amount
                elif old_status == "approved" and mp_status in ("cancelled", "refunded", "charged_back"):
                    current_user.credits_balance = max(0.0, current_user.credits_balance - payment.credits_amount)
                db.commit()
                db.refresh(payment)
    except Exception:
        pass
    return PaymentOut.model_validate(payment)
