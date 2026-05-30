#!/usr/bin/env python3
"""Script para executar migração SQL no banco de dados"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def run_migration():
    database_url = os.getenv("DATABASE_URL", "")
    
    if not database_url:
        print("❌ DATABASE_URL não configurada")
        return False
    
    print("🔄 Conectando ao banco de dados...")
    print(f"   Host: {database_url.split('@')[1].split('/')[0] if '@' in database_url else 'N/A'}")
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            print("📝 Executando migração...")
            
            # Adicionar coluna auto_reply_enabled
            conn.execute(text("""
                ALTER TABLE channels 
                ADD COLUMN IF NOT EXISTS auto_reply_enabled BOOLEAN DEFAULT FALSE
            """))
            
            # Adicionar coluna auto_reply_prompt
            conn.execute(text("""
                ALTER TABLE channels 
                ADD COLUMN IF NOT EXISTS auto_reply_prompt TEXT
            """))
            
            # Update existing channels
            conn.execute(text("""
                UPDATE channels 
                SET auto_reply_enabled = FALSE 
                WHERE auto_reply_enabled IS NULL
            """))
            
            conn.commit()
            
            print("✅ Migração executada com sucesso!")
            
            # Verificar se as colunas foram adicionadas
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'channels' 
                AND column_name IN ('auto_reply_enabled', 'auto_reply_prompt')
                ORDER BY column_name
            """))
            
            columns = result.fetchall()
            print(f"\n📊 Colunas adicionadas ({len(columns)}):")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} (nullable: {col[2]}, default: {col[3]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao executar migração: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_migration()
    exit(0 if success else 1)
