import json
import os
from datetime import datetime, timedelta

from aje_libs.common.helpers.bedrock_helper import BedrockHelper
from aje_libs.common.helpers.dynamodb_helper import DynamoDBHelper
from aje_libs.common.helpers.ssm_helper import SSMParameterHelper
from aje_libs.common.logger import custom_logger

# Configuración
ENVIRONMENT = os.environ["ENVIRONMENT"]
ENTERPRISE = os.environ["ENTERPRISE"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]
CHALLENGE_FEEDBACK_HISTORY_TABLE = os.environ["CHALLENGE_FEEDBACK_HISTORY_TABLE"]

# Parameter Store
ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/agent")
PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())
LLM_MODEL_ID = PARAMETER_VALUE["LLM_MODEL_ID"]
LLM_REGION = PARAMETER_VALUE["LLM_REGION"]
LLM_MAX_TOKENS = int(PARAMETER_VALUE["LLM_MAX_TOKENS"])

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

challenge_feedback_table_helper = DynamoDBHelper(
    table_name=CHALLENGE_FEEDBACK_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="reto_resultado_id"
)

bedrock_helper = BedrockHelper(region_name=LLM_REGION)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

GENERAL_FEEDBACK_PROMPT = """
## Rol
Eres un evaluador académico experto en {nombre_curso}. Debes redactar una retroalimentación final, integral y personalizada para un estudiante que ha completado un reto de evaluación.

---

## Contexto del reto
- Curso: {nombre_curso}
- Complejidad: {complejidad}
- Intentos realizados: {numero_intentos}

---

## Criterio de logro

- Umbral de logro: {umbral}

IMPORTANTE:
- El nivel esperado se alcanza cuando el estudiante demuestra un desempeño alto en la mayoría de los criterios.
- Si predominan niveles "medio" o "bajo", se considera que no se alcanzó el nivel esperado.
- Basa esta conclusión únicamente en los niveles de desempeño observados.

---

## Criterios de evaluación

{criterios_formateados}

IMPORTANTE:
- Cada criterio incluye los temas clave que debían aparecer en las respuestas.
- Úsalos como referencia para identificar vacíos conceptuales específicos.

---

## Historial de intentos

{intentos_formateados}

IMPORTANTE:
- Analiza la evolución del estudiante entre intentos.
- Detecta si mejoró, se estancó o repitió los mismos errores.
- Identifica qué criterios se mantuvieron en nivel "bajo".
- Usa los niveles cualitativos por criterio como evidencia objetiva.

---

## Instrucciones

1. ANALIZA la progresión del estudiante considerando:
   - Calidad de las respuestas en cada intento.
   - Evolución (o ausencia de ella) por criterio.
   - Feedbacks previos y si el estudiante los incorporó.

2. REDACTA un párrafo único con las siguientes características:
   - Segunda persona del singular (tú).
   - Describe cualitativamente el nivel de comprensión: usa expresiones como "lograste demostrar", "tu comprensión fue parcial", "no llegaste a articular".
   - Menciona explícitamente los criterios o temas donde hubo debilidad persistente.
   - Si hubo mejora entre intentos, reconócela de forma natural sin mencionar números de intento.
   - Integra los errores como sugerencias de mejora constructivas.
   - Incluye una recomendación concreta orientada a reforzar los criterios más débiles.
   - Expresa cualitativamente si se alcanzó o no el nivel esperado: "lograste alcanzar el nivel esperado" o "no llegaste a alcanzar el nivel requerido".

---

## Restricciones absolutas
- NO menciones scores, puntajes, porcentajes ni valores numéricos.
- NO hagas referencia a número de intentos.
- NO uses lenguaje genérico (ej. "debes mejorar", "buen trabajo").
- NO uses signos de exclamación ni lenguaje coloquial.
- NO uses listas, encabezados ni saltos de formato.
- NO inventes conocimientos no evidenciados en la respuesta del estudiante.
- LIMÍTATE a un solo párrafo continuo.

---

## Validación interna antes de responder
- Se menciona al menos un concepto o tema débil de forma explícita.
- Se incluye una recomendación concreta.
- Se indica si alcanzó o no el nivel esperado.
- No hay ningún número en el texto.
- Es un solo párrafo continuo.

---

## Formato de salida (obligatorio)

Responde ÚNICAMENTE con un JSON válido:

{{
  "feedback": "Retroalimentación final en segunda persona, en un solo párrafo continuo"
}}
"""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _format_criterios(criterios: list) -> str:
    """
    Formatea criterios en estructura semi-estructurada para mejor parsing del LLM.
    """
    blocks = []
    for crit in criterios:
        temas = [campo["CampoTematico"] for campo in crit.get("CamposTematicos", [])]
        block = (
            f"- criterio_id: {crit['CriterioId']}\n"
            f"  descripcion: {crit['Criterio']}\n"
            f"  temas: [{', '.join(temas)}]"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _format_intentos(intentos: list, criterios: list, umbral: float) -> str:
    """
    Formatea el historial de intentos enriquecido con nivel cualitativo por criterio.
    Incluye el fragmento del caso de cada intento.
    No expone scores numéricos al LLM.
    """
    criterios_map = {crit["CriterioId"]: crit["Criterio"] for crit in criterios}
    limite_medio  = round(umbral * 0.7, 2)
    blocks = []

    for intento in intentos:
        scores = intento.get("Scores", [])
        niveles_lines = []
        for sc in scores:
            crit_id   = sc["CriterioId"]
            score_val = float(sc["Score"])
            nivel     = "alto" if score_val >= umbral else "medio" if score_val >= limite_medio else "bajo"
            descripcion = criterios_map.get(crit_id, f"criterio {crit_id}")
            niveles_lines.append(f"    · {descripcion}: {nivel}")

        block = (
            f"- intento: {intento.get('NumeroIntento', '?')}\n"
            f"  titulo: {intento.get('Titulo', '')}\n"
            f"  pregunta: {intento.get('Pregunta', '')}\n"
            f"  respuesta_modelo: {intento.get('RespuestaModelo', '')}\n"
            f"  respuesta_estudiante: {intento.get('RespuestaUsuario', '')}\n"
            f"  desempeño_por_criterio:\n" + "\n".join(niveles_lines) + "\n"
            f"  feedback_recibido: {intento.get('Feedback', '')}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def _parse_llm_response(
    raw_text: str
) -> dict:

    clean = raw_text.strip()

    clean = clean.removeprefix("```json")
    clean = clean.removeprefix("```")
    clean = clean.removesuffix("```")
    clean = clean.strip()

    return json.loads(clean)


def _invoke_prompt(prompt: str, max_tokens: int, temperature: float = 0.0, top_p: float = 0.2) -> dict:
    response = bedrock_helper.converse(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        parameters={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
    )
    logger.info(f"Respuesta del modelo: {response}")
    return response


def _upload_feedback(
    usuario_id: int,
    reto_resultado_id: str,
    silabo_id: int,
    unidad_id: int,
    sesion_id: int,
    prompt: str,
    ai_result: str,
    input_tokens: int,
    output_tokens: int
):
    try:
        ttl_timestamp = int((datetime.now() + timedelta(seconds=432000)).timestamp())  # TTL 5 días

        item = {
            "usuario_id": usuario_id,
            "reto_resultado_id": reto_resultado_id,
            "tipo_metodo_id": 674, # Ruta estándar
            "silabo_id": silabo_id,
            "unidad_id": unidad_id,
            "sesion_id": sesion_id,
            "prompt": prompt,
            "ai_result": ai_result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ttl": ttl_timestamp
        }

        challenge_feedback_table_helper.put_item(data=item)
        logger.info(f"Retroalimentación general persistida: usuario_id={usuario_id}, reto_resultado_id={reto_resultado_id}")

    except Exception as e:
        logger.error(f"Error al persistir retroalimentación general: {e}")


# ─────────────────────────────────────────────
# VALIDACIONES
# ─────────────────────────────────────────────

def _validate_criterios(criterios: list) -> str | None:
    if not isinstance(criterios, list) or len(criterios) == 0:
        return "El campo 'Criterios' debe ser una lista no vacía."

    for crit in criterios:
        if not all(k in crit for k in ("CriterioId", "Criterio", "CamposTematicos")):
            return "Cada criterio debe contener 'CriterioId', 'Criterio' y 'CamposTematicos'."

        campos = crit.get("CamposTematicos", [])
        if not isinstance(campos, list) or len(campos) == 0:
            return "El campo 'CamposTematicos' dentro de cada criterio debe ser una lista no vacía."

        for campo in campos:
            if not all(k in campo for k in ("CampoTematicoId", "CampoTematico")):
                return "Cada campo temático debe contener 'CampoTematicoId' y 'CampoTematico'."

    return None


def _validate_intentos(intentos: list) -> str | None:
    if not isinstance(intentos, list) or len(intentos) == 0:
        return "El campo 'Intentos' debe ser una lista no vacía."

    for intento in intentos:
        required = ("NumeroIntento", "Titulo", "Pregunta", "RespuestaModelo", "RespuestaUsuario", "Feedback", "Scores")
        if not all(k in intento for k in required):
            return f"Cada intento debe contener: {', '.join(required)}."

        scores = intento.get("Scores", [])
        if not isinstance(scores, list) or len(scores) == 0:
            return "El campo 'Scores' dentro de cada intento debe ser una lista no vacía."

        for sc in scores:
            if not all(k in sc for k in ("CriterioId", "Score")):
                return f"Cada elemento de 'Scores' en el intento {intento.get('NumeroIntento', '?')} debe contener 'CriterioId' y 'Score'."

    return None


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        logger.info(f"Event path general feedback: {event}")

        body = event.get("body")
        if not body:
            return {"statusCode": 400, "body": json.dumps({"success": False, "message": "Body requerido"})}
        body = json.loads(body) if isinstance(body, str) else body

        # ── Validar campos de primer nivel ──
        required_fields = [
            "UsuarioId", "SilaboId", "UnidadId", "SesionId", "RetoResultadoId",
            "NombreCurso", "Complejidad", "Umbral", "MaximosIntentos",
            "Criterios", "Intentos"
        ]
        missing_fields = [f for f in required_fields if f not in body]
        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": f"Campos requeridos faltantes: {missing_fields}"})
            }

        # ── Validar criterios ──
        validation_error = _validate_criterios(body["Criterios"])
        if validation_error:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": validation_error})
            }

        # ── Validar intentos ──
        validation_error = _validate_intentos(body["Intentos"])
        if validation_error:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": validation_error})
            }

        # ── Extraer campos ──
        usuario_id        = body["UsuarioId"]
        silabo_id         = body["SilaboId"]
        unidad_id         = body["UnidadId"]
        sesion_id         = body["SesionId"]
        reto_resultado_id = body["RetoResultadoId"]
        nombre_curso      = body["NombreCurso"]
        complejidad       = body["Complejidad"]
        umbral            = body["Umbral"]
        maximos_intentos  = body["MaximosIntentos"]
        criterios         = body["Criterios"]
        intentos          = body["Intentos"]
        numero_intentos   = len(intentos)

        criterios_formateados = _format_criterios(criterios)
        intentos_formateados  = _format_intentos(intentos, criterios, umbral)  # <-- recibe criterios y umbral para enriquecer

        # ── PASO 1: Invocar modelo ──
        prompt = GENERAL_FEEDBACK_PROMPT.format(
            nombre_curso=nombre_curso,
            complejidad=complejidad,
            umbral=umbral,
            numero_intentos=numero_intentos,
            criterios_formateados=criterios_formateados,
            intentos_formateados=intentos_formateados,
        )

        response      = _invoke_prompt(prompt=prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.5, top_p=0.9)
        raw_result    = response["output"]["message"]["content"][0]["text"]
        input_tokens  = response["usage"]["inputTokens"]
        output_tokens = response["usage"]["outputTokens"]

        # ── PASO 2: Persistir en DynamoDB (siempre) ──
        _upload_feedback(
            usuario_id=usuario_id,
            reto_resultado_id=reto_resultado_id,
            silabo_id=silabo_id,
            unidad_id=unidad_id,
            sesion_id=sesion_id,
            prompt=prompt,
            ai_result=raw_result,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # ── PASO 3: Parsear respuesta ──
        try:
            feedback_result = _parse_llm_response(raw_result)
        except json.JSONDecodeError:
            return {
                "statusCode": 502,
                "body": json.dumps({
                    "success": False,
                    "message": "El modelo no retornó el feedback en formato esperado.",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
            }

        # ── PASO 4: Construir respuesta ──
        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "feedback": feedback_result.get("feedback"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }

    except Exception as e:
        logger.error(f"Error in path general feedback: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"success": False, "message": str(e)})
        }