import json
import os
from datetime import datetime, timedelta

from aje_libs.common.helpers.bedrock_helper import BedrockHelper
from aje_libs.common.helpers.dynamodb_helper import DynamoDBHelper
from aje_libs.common.helpers.ssm_helper import SSMParameterHelper
from aje_libs.common.logger import custom_logger

# Configuración
ENVIRONMENT = os.environ["ENVIRONMENT"]
ENTERPRISE  = os.environ["ENTERPRISE"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
OWNER = os.environ["OWNER"]
CHALLENGE_EVALUATION_HISTORY_TABLE = os.environ["CHALLENGE_EVALUATION_HISTORY_TABLE"]

# Parameter Store
ssm_agent = SSMParameterHelper(f"/{ENVIRONMENT}/{PROJECT_NAME}/{ENTERPRISE}/agent")
PARAMETER_VALUE = json.loads(ssm_agent.get_parameter_value())
LLM_MODEL_ID   = PARAMETER_VALUE["LLM_MODEL_ID"]
LLM_REGION     = PARAMETER_VALUE["LLM_REGION"]
LLM_MAX_TOKENS = int(PARAMETER_VALUE["LLM_MAX_TOKENS"])

logger = custom_logger(__name__, owner=OWNER, service=PROJECT_NAME)

challenge_evaluation_table_helper = DynamoDBHelper(
    table_name=CHALLENGE_EVALUATION_HISTORY_TABLE,
    pk_name="usuario_id",
    sk_name="reto_iteracion_id"
)

bedrock_helper = BedrockHelper(region_name=LLM_REGION)

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

# El score es una medición objetiva de 0 a 1, independiente del umbral.
SCORE_PROMPT = """
## Rol
Eres un evaluador académico experto en {nombre_curso}. Debes calificar objetivamente la respuesta de un estudiante comparándola con una respuesta modelo.

---

## Fragmento del caso
{contexto}

---

## Pregunta
{pregunta}

---

## Respuesta del estudiante
{respuesta_usuario}

---

## Respuesta modelo
{respuesta_modelo}

---

## Criterios de evaluación

{criterios_formateados}

IMPORTANTE:
- Cada criterio incluye "temas", que representan los conceptos clave que deben aparecer explícitamente en la respuesta.
- Evalúa SOLO lo que el estudiante escribió considerando el fragmento del caso como contexto (no infieras conocimiento implícito).
- Si un tema no está mencionado o explicado, debe penalizarse el puntaje.

---

## Instrucciones de evaluación

Evalúa cada criterio de forma INDEPENDIENTE considerando:

1. Precisión conceptual
2. Cobertura de los temas clave
3. Equivalencia semántica con la respuesta modelo (solo en lo relevante al criterio)
4. Correcta aplicación del fragmento del caso al criterio evaluado

---

## Escala de puntuación (usa estos anclajes)

- 1.00 → Explicación completa, precisa y equivalente al modelo
- 0.75 → Mayormente correcta, con omisiones menores
- 0.50 → Comprensión parcial (idea general sin detalles clave)
- 0.25 → Mención superficial o vaga
- 0.00 → Sin contenido relevante

Puedes usar valores intermedios (ej. 0.60, 0.40) si es necesario.

---

## Reglas estrictas

- Evalúa cada criterio por separado.
- Penaliza explícitamente la falta de mención o explicación de los temas clave asociados a cada criterio.
- No otorgues puntaje alto si solo hay definiciones generales sin aplicación al caso.
- Si la respuesta es irrelevante, vacía o ruido, asigna 0.00 a TODOS los criterios.
- Solo asigna puntaje > 0.00 si hay contenido académico relacionado con ese criterio.
- No expliques tu evaluación.

---

## Validación antes de responder

- Verifica que todos los criterios tengan un score.
- Verifica que los criterio_id coincidan EXACTAMENTE con los proporcionados.
- Verifica que todos los scores estén entre 0.00 y 1.00.

---

## Formato de salida (obligatorio)

Responde ÚNICAMENTE con un JSON válido:

{{
  "scores_criterios": [
    {{"criterio_id": <CriterioId>, "score": <valor entre 0.00 y 1.00>}},
    ...
  ]
}}
"""

# Prompt para respuesta INCORRECTA: genera feedback + nuevo reto en JSON
FEEDBACK_ALL_PROMPT = """
## Rol
Eres un evaluador académico experto en {nombre_curso}. Debes generar retroalimentación precisa y una nueva oportunidad de aprendizaje contextualizada en el caso clínico.

---

## Contexto
- Curso: {nombre_curso}
- Reto: {titulo_reto}
- Complejidad: {complejidad}

Fragmento del caso:
{contexto}

Pregunta:
{pregunta}

Respuesta modelo:
{respuesta_modelo}

Respuesta del estudiante:
{respuesta_usuario}

---

## Evaluación por criterios

{criterios_con_scores}

IMPORTANTE:
- "nivel" indica el logro respecto al umbral de aprobación ({umbral}):
  • alto   → score >= {umbral} (dominio adecuado)
  • medio  → score por debajo del umbral pero con comprensión parcial
  • bajo   → comprensión insuficiente
- Basa TODA la retroalimentación en estos niveles.
- No ignores criterios con nivel bajo o medio.

{intentos_anteriores_section}

## Instrucciones

### 1. Retroalimentación (máximo 3 líneas)
- Dirígete al estudiante en segunda persona (tú).
- Señala explícitamente los criterios con nivel "bajo" (score < {limite_medio}) y qué faltó (ej. mecanismos, relaciones, procesos).
- Si hay logros parciales (nivel "medio"), menciona qué idea sí fue correcta pero qué faltó para alcanzar el umbral.
- Si hay patrón en intentos anteriores, intégralo de forma natural.
- Sé específico: menciona conceptos concretos del caso (no generalidades).

---

### 2. Nuevo reto
- Debe enfocarse en los criterios con nivel "bajo" o "medio".
- No repitas el enfoque de preguntas anteriores.
- Si el desempeño fue bajo (score < {limite_medio}), simplifica ligeramente la complejidad.
- Si fue parcial (nivel "medio"), mantén nivel pero enfoca en lo omitido.
- La pregunta DEBE estar contextualizada en el fragmento del caso: usa únicamente las situaciones, actores o datos que aparecen en ese fragmento. No inventes ni añadas información del caso que no esté presente.
- Debe ser una pregunta abierta que obligue a explicar procesos aplicados al caso (no definiciones aisladas).

---

## Validación interna antes de responder
- El feedback menciona al menos un criterio débil con referencia al caso.
- El reto usa datos del fragmento del caso y no inventa información nueva.
- No repite la pregunta original ni las de intentos anteriores.

---

## Formato de salida (obligatorio)

Responde ÚNICAMENTE con un JSON válido:

{{
  "feedback": "Texto en máximo 3 líneas",
  "reto": {{
    "titulo": "Máximo 6 palabras",
    "pregunta": "Pregunta abierta contextualizada en el caso",
    "respuesta_modelo": "Respuesta clara, técnica y completa"
  }}
}}
"""

# Prompt para respuesta CORRECTA: solo feedback en JSON
FEEDBACK_PROMPT = """
## Rol
Eres un evaluador académico experto en {nombre_curso}. Debes generar retroalimentación precisa para un estudiante con buen desempeño.

---

## Contexto
- Curso: {nombre_curso}
- Reto: {titulo_reto}
- Complejidad: {complejidad}

Fragmento del caso:
{contexto}

Pregunta:
{pregunta}

Respuesta modelo:
{respuesta_modelo}

Respuesta del estudiante:
{respuesta_usuario}

---

## Evaluación por criterios

{criterios_con_scores}

IMPORTANTE:
- Considera "nivel" como evidencia objetiva del desempeño respecto al umbral ({umbral}).
- Reconoce SOLO lo que está reflejado en los niveles obtenidos.

{intentos_anteriores_section}

## Instrucciones

- Redacta en segunda persona (tú).
- Reconoce explícitamente los criterios con nivel "alto" (score >= {umbral}).
- Si hay criterios con nivel "medio" (score entre {limite_medio} y {umbral}), menciona brevemente qué podría profundizar aplicado al caso.
- Si hay evolución respecto a intentos anteriores, intégrala naturalmente.
- Incluye UNA sugerencia concreta para profundizar referenciando el caso (no genérica).
- Mantén tono académico, preciso y sin expresiones emocionales.
- Máximo 3 líneas, un solo párrafo.

---

## Validación interna
- Se mencionan criterios logrados.
- La sugerencia es específica al caso y al criterio (no genérica).
- No hay lenguaje genérico.

---

## Formato de salida (obligatorio)

Responde ÚNICAMENTE con un JSON válido:

{{
  "feedback": "Texto en máximo 3 líneas"
}}
"""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_intentos_anteriores_section(intentos_anteriores: list, umbral: float, limite_medio: float) -> str:
    """
    Formato estructurado para análisis de progresión por el LLM.
    Expone niveles cualitativos en lugar de scores numéricos.
    """
    if not intentos_anteriores:
        return ""

    blocks = ["---\n\n## Historial de intentos del estudiante"]

    for intento in intentos_anteriores:
        scores = intento.get("Scores", [])
        niveles_str = ", ".join([
            f"{sc['CriterioId']}:{'alto' if float(sc['Score']) >= umbral else 'medio' if float(sc['Score']) >= limite_medio else 'bajo'}"
            for sc in scores
        ])

        block = (
            f"- intento: {intento.get('NumeroIntento', '?')}\n"
            f"  titulo: {intento.get('Titulo', '')}\n"
            f"  pregunta: {intento.get('Pregunta', '')}\n"
            f"  respuesta_modelo: {intento.get('RespuestaModelo', '')}\n"
            f"  respuesta_estudiante: {intento.get('RespuestaUsuario', '')}\n"
            f"  niveles: [{niveles_str}]\n"
            f"  feedback: {intento.get('Feedback', '')}"
        )
        blocks.append(block)

    blocks.append("""
IMPORTANTE:
- Identifica patrones de error repetidos en los intentos.
- Detecta si el estudiante mejora, se estanca o repite errores.
- Usa esta información para hacer la retroalimentación más precisa.
""")

    return "\n\n".join(blocks) + "\n\n---"


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


def _format_criterios_con_scores(criterios: list, scores_criterios: list, umbral: float, limite_medio: float) -> str:
    """
    Formato estructurado para LLM. Sin scores numéricos, solo niveles cualitativos.
    """
    scores_map = {sc["criterio_id"]: sc["score"] for sc in scores_criterios}
    blocks = []
    for crit in criterios:
        crit_id = crit["CriterioId"]
        score_val = scores_map.get(crit_id, 0.0)
        nivel = "alto" if score_val >= umbral else "medio" if score_val >= limite_medio else "bajo"
        temas = [c["CampoTematico"] for c in crit.get("CamposTematicos", [])]
        block = (
            f"- criterio_id: {crit_id}\n"
            f"  descripcion: {crit['Criterio']}\n"
            f"  nivel: {nivel}\n"
            f"  temas: [{', '.join(temas)}]"
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


def _upload_evaluation(
    usuario_id: int,
    reto_iteracion_id: str,
    silabo_id: int,
    unidad_id: int,
    sesion_id: int,
    score_prompt: str,       # <-- nuevo
    score_raw: str,          # <-- nuevo
    feedback_prompt: str,
    feedback_raw: str,
    input_tokens: int,
    output_tokens: int
):
    try:
        ttl_timestamp = int((datetime.now() + timedelta(seconds=432000)).timestamp())

        item = {
            "usuario_id":        usuario_id,
            "reto_iteracion_id": reto_iteracion_id,
            "tipo_metodo_id":    675, # Ruta con caso
            "silabo_id":         silabo_id,
            "unidad_id":         unidad_id,
            "sesion_id":         sesion_id,
            "score_prompt":      score_prompt,
            "score_raw":         score_raw,
            "feedback_prompt":   feedback_prompt,
            "feedback_raw":      feedback_raw,
            "input_tokens":      input_tokens,
            "output_tokens":     output_tokens,
            "date_time":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ttl":               ttl_timestamp
        }

        challenge_evaluation_table_helper.put_item(data=item)
        logger.info(f"Evaluación persistida: usuario_id={usuario_id}, reto_iteracion_id={reto_iteracion_id}")

    except Exception as e:
        logger.error(f"Error al persistir evaluación: {e}")


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


def _validate_intentos_anteriores(intentos: list) -> str | None:
    if not isinstance(intentos, list):
        return "El campo 'IntentosAnteriores' debe ser una lista."

    for intento in intentos:
        required = ("NumeroIntento", "Titulo", "Pregunta", "RespuestaModelo", "RespuestaUsuario", "Feedback", "Scores")
        if not all(k in intento for k in required):
            return f"Cada intento anterior debe contener: {', '.join(required)}."

        scores = intento.get("Scores", [])
        if not isinstance(scores, list) or len(scores) == 0:
            return "El campo 'Scores' dentro de cada intento anterior debe ser una lista no vacía."

        for sc in scores:
            if not all(k in sc for k in ("CriterioId", "Score")):
                return f"Cada elemento de 'Scores' en el intento {intento.get('NumeroIntento', '?')} debe contener 'CriterioId' y 'Score'."

    return None


def _validate_scores_criterios(scores_criterios: list, criterios: list) -> str | None:
    criterio_ids_esperados = {crit["CriterioId"] for crit in criterios}
    criterio_ids_recibidos = {sc.get("criterio_id") for sc in scores_criterios}

    faltantes = criterio_ids_esperados - criterio_ids_recibidos
    if faltantes:
        return f"El modelo no retornó scores para los siguientes criterios: {faltantes}"

    for sc in scores_criterios:
        score_val = sc.get("score")
        if not isinstance(score_val, (int, float)) or not (0.0 <= float(score_val) <= 1.0):
            return f"Score inválido para criterio_id {sc.get('criterio_id')}: '{score_val}'. Debe ser un número entre 0.0 y 1.0."

    return None


# ─────────────────────────────────────────────
# HANDLER
# ─────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        logger.info(f"Event case path evaluate: {event}")

        body = event.get("body")
        if not body:
            return {"statusCode": 400, "body": json.dumps({"success": False, "message": "Body requerido"})}
        body = json.loads(body) if isinstance(body, str) else body

        # ── Validar campos de primer nivel ──
        required_fields = [
            "UsuarioId", "SilaboId", "UnidadId", "SesionId", "RetoIteracionId",
            "NombreCurso", "Complejidad", "Umbral", "Reto", "IntentosAnteriores"
        ]
        missing_fields = [f for f in required_fields if f not in body]
        if missing_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": f"Campos requeridos faltantes: {missing_fields}"})
            }

        # ── Validar estructura de Reto ──
        reto = body["Reto"]
        required_reto_fields = ["Titulo", "Contexto", "Pregunta", "RespuestaModelo", "RespuestaUsuario", "Criterios"]
        missing_reto_fields = [f for f in required_reto_fields if f not in reto]
        if missing_reto_fields:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": f"Campos requeridos faltantes en 'Reto': {missing_reto_fields}"})
            }

        if not isinstance(reto.get("Contexto", ""), str) or not reto["Contexto"].strip():
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": "El campo 'Contexto' dentro de 'Reto' debe ser un texto no vacío."})
            }

        validation_error = _validate_criterios(reto["Criterios"])
        if validation_error:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": validation_error})
            }

        validation_error = _validate_intentos_anteriores(body["IntentosAnteriores"])
        if validation_error:
            return {
                "statusCode": 400,
                "body": json.dumps({"success": False, "message": validation_error})
            }

        # ── Extraer campos ──
        reto_iteracion_id   = body["RetoIteracionId"]
        usuario_id          = body["UsuarioId"]
        silabo_id           = body["SilaboId"]
        unidad_id           = body["UnidadId"]
        sesion_id           = body["SesionId"]
        nombre_curso        = body["NombreCurso"]
        complejidad         = body["Complejidad"]
        umbral              = body["Umbral"]
        intentos_anteriores = body["IntentosAnteriores"]

        titulo_reto         = reto["Titulo"]
        contexto            = reto["Contexto"]
        pregunta            = reto["Pregunta"]
        respuesta_modelo    = reto["RespuestaModelo"]
        respuesta_usuario   = reto["RespuestaUsuario"]
        criterios           = reto["Criterios"]

        criterios_formateados = _format_criterios(criterios)

        # ── PASO 1: Obtener scores por criterio ──────────────────────────────
        score_prompt = SCORE_PROMPT.format(
            nombre_curso=nombre_curso,
            contexto=contexto,
            pregunta=pregunta,
            respuesta_usuario=respuesta_usuario,
            respuesta_modelo=respuesta_modelo,
            criterios_formateados=criterios_formateados,
        )

        score_response = _invoke_prompt(prompt=score_prompt, max_tokens=256)
        score_raw = score_response["output"]["message"]["content"][0]["text"]

        try:
            score_result = _parse_llm_response(score_raw)
            scores_criterios = score_result.get("scores_criterios", [])
        except json.JSONDecodeError:
            _upload_evaluation(
                usuario_id=usuario_id,
                reto_iteracion_id=reto_iteracion_id,
                silabo_id=silabo_id,
                unidad_id=unidad_id,
                sesion_id=sesion_id,
                score_prompt=score_prompt,
                score_raw=score_raw,
                feedback_prompt="",
                feedback_raw="",
                input_tokens=0,
                output_tokens=0
            )
            return {
                "statusCode": 502,
                "body": json.dumps({
                    "success": False,
                    "message": "El modelo no retornó los scores en formato JSON esperado."
                })
            }

        validation_error = _validate_scores_criterios(scores_criterios, criterios)
        if validation_error:
            _upload_evaluation(
                usuario_id=usuario_id,
                reto_iteracion_id=reto_iteracion_id,
                silabo_id=silabo_id,
                unidad_id=unidad_id,
                sesion_id=sesion_id,
                score_prompt=score_prompt,
                score_raw=score_raw,
                feedback_prompt="",
                feedback_raw="",
                input_tokens=0,
                output_tokens=0
            )
            return {
                "statusCode": 502,
                "body": json.dumps({
                    "success": False,
                    "message": f"Respuesta del modelo inválida: {validation_error}"
                })
            }

        # Score promedio para comparar con el umbral (uso interno, no se retorna)
        score_promedio = round(
            sum(float(sc["score"]) for sc in scores_criterios) / len(scores_criterios), 2
        )

        # ── PASO 2: Generar feedback (y nuevo reto si aplica) ────────────────
        limite_medio = round(umbral * 0.7, 2)
        criterios_con_scores = _format_criterios_con_scores(criterios, scores_criterios, umbral, limite_medio)
        intentos_anteriores_section = _build_intentos_anteriores_section(intentos_anteriores, umbral, limite_medio)

        if score_promedio < umbral:
            feedback_prompt = FEEDBACK_ALL_PROMPT.format(
                nombre_curso=nombre_curso,
                titulo_reto=titulo_reto,
                complejidad=complejidad,
                contexto=contexto,
                pregunta=pregunta,
                respuesta_modelo=respuesta_modelo,
                respuesta_usuario=respuesta_usuario,
                criterios_con_scores=criterios_con_scores,
                umbral=umbral,
                limite_medio=limite_medio,
                intentos_anteriores_section=intentos_anteriores_section
            )
        else:
            feedback_prompt = FEEDBACK_PROMPT.format(
                nombre_curso=nombre_curso,
                titulo_reto=titulo_reto,
                complejidad=complejidad,
                contexto=contexto,
                pregunta=pregunta,
                respuesta_modelo=respuesta_modelo,
                respuesta_usuario=respuesta_usuario,
                criterios_con_scores=criterios_con_scores,
                umbral=umbral,
                limite_medio=limite_medio,
                intentos_anteriores_section=intentos_anteriores_section
            )

        feedback_response = _invoke_prompt(
            prompt=feedback_prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
            top_p=0.9
        )
        feedback_raw  = feedback_response["output"]["message"]["content"][0]["text"]
        input_tokens  = feedback_response["usage"]["inputTokens"]
        output_tokens = feedback_response["usage"]["outputTokens"]

        # ── PASO 3: Persistir en DynamoDB ────────────────────────────────────
        _upload_evaluation(
            usuario_id=usuario_id,
            reto_iteracion_id=reto_iteracion_id,
            silabo_id=silabo_id,
            unidad_id=unidad_id,
            sesion_id=sesion_id,
            score_prompt=score_prompt,
            score_raw=score_raw,
            feedback_prompt=feedback_prompt,
            feedback_raw=feedback_raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # ── PASO 4: Parsear feedback ─────────────────────────────────────────
        try:
            feedback_result = _parse_llm_response(feedback_raw)
        except json.JSONDecodeError:
            return {
                "statusCode": 502,
                "body": json.dumps({
                    "success": False,
                    "message": "El modelo no retornó el feedback en formato esperado.",
                    "scores_criterios": scores_criterios,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
            }

        # ── PASO 5: Construir respuesta ──────────────────────────────────────
        return {
            "statusCode": 200,
            "body": json.dumps({
                "success": True,
                "scores_criterios": scores_criterios,
                "feedback": feedback_result.get("feedback"),
                "reto": feedback_result.get("reto", {}),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
        }

    except Exception as e:
        logger.error(f"Error in case path evaluate: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"success": False, "message": str(e)})
        }