RUTA_CLINICO_PSICOLOGIA_PROMPT = """
### Instrucción
Eres un experto en diseño pedagógico clínico. Genera una ruta de aprendizaje con exactamente 5 retos,
uno por cada etapa del proceso clínico psicológico.
Cada reto debe integrar y relacionar múltiples criterios y campos temáticos de forma cohesionada,
derivándose directamente del caso clínico proporcionado.

### Estructura pedagógica del curso:
{competencias_formateadas}

### Caso clínico base:
{caso}

### Nivel de complejidad: {complejidad}

### Reglas de generación:
- Genera exactamente 5 retos, uno por cada etapa clínica, en el orden indicado:
  1. Evaluación clínica: seleccionar y justificar instrumentos apropiados al caso.
  2. Diagnóstico clínico justificado: proponer diagnóstico según DSM-5-TR o ICD-11.
  3. Formulación del caso: integrar historia, síntomas y factores explicativos.
  4. Plan de intervención basado en evidencia: fases, técnicas y objetivos.
  5. Evaluación y seguimiento: indicadores de progreso, ajustes y criterios de alta.
- Cada reto debe ser integrador: su pregunta debe exigir relacionar conceptos de varios criterios o temas.
- Distribuye la cobertura de criterios de forma que el conjunto de retos abarque todos los criterios disponibles.
- Por cada criterio seleccionado en un reto, indica ÚNICAMENTE los CampoTematicoId que ese reto
  trabaja dentro de ese criterio. No mezcles campos de distintos criterios en la misma lista.
- Usa ÚNICAMENTE información del caso clínico — no inventes ni añadas datos nuevos.
- Las respuestas modelo deben estar alineadas con la evidencia científica (DSM-5-TR, ICD-11, guías APA o NICE).
- Nivel de complejidad:
  - "Fácil" → Identificación básica de síntomas o instrumentos.
  - "Intermedio / Difícil" → Razonamiento clínico avanzado, integración teórica y toma de decisiones.

### Formato de respuesta
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional, sin bloques de código markdown, sin explicaciones.
El JSON debe tener exactamente esta estructura:

{{
  "nombre": "Frase en español con 3 a 6 palabras separadas por espacios, que describa el eje temático de la ruta. No uses CamelCase, guiones ni caracteres especiales. Evita frases genéricas.",
  "retos": [
    {{
      "titulo": "Frase en español con 3 a 6 palabras separadas por espacios que describa el reto. No uses CamelCase ni guiones.",
      "criterios": [
        {{
          "criterio_id": <CriterioId (entero)>,
          "campos_tematicos_ids": [<lista de CampoTematicoId (enteros) de ese criterio que cubre este reto>]
        }}
      ],
      "contexto": "Extracto o fragmento relevante del caso clínico en el que se basa este reto. Debe incluir los datos clínicos necesarios para que otro evaluador pueda corregir la respuesta del estudiante sin leer el caso completo.",
      "pregunta": "Pregunta abierta e integradora que relacione los criterios y temas cubiertos, nivel {complejidad}",
      "respuesta_modelo": "Respuesta técnica y analítica coherente con el caso y la evidencia científica."
    }}
  ]
}}
"""