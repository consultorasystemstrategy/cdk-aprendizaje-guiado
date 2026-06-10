RUTA_ESTRATEGICO_PROMPT = """
### Instrucción
Eres un experto en diseño pedagógico. Genera una ruta de aprendizaje con exactamente 5 retos,
uno por cada etapa del análisis de caso estratégico.
Cada reto debe integrar y relacionar múltiples criterios y campos temáticos de forma cohesionada.

### Estructura pedagógica del curso:
{competencias_formateadas}

### Caso base:
{caso}

### Nivel de complejidad: {complejidad}

### Reglas de generación:
- Genera exactamente 5 retos, uno por cada etapa, en el orden indicado:
  1. Identificación del problema central.
  2. Análisis causal y diagnóstico.
  3. Generación de alternativas.
  4. Evaluación y selección.
  5. Plan de acción.
- Cada reto debe ser integrador: su pregunta debe exigir relacionar conceptos de varios criterios o temas.
- Distribuye la cobertura de criterios de forma que el conjunto de retos abarque todos los criterios disponibles.
- Por cada criterio seleccionado en un reto, indica ÚNICAMENTE los CampoTematicoId que ese reto
  trabaja dentro de ese criterio. No mezcles campos de distintos criterios en la misma lista.
- Basa los retos en el caso proporcionado y en los criterios y campos temáticos de la estructura pedagógica.
- Nivel de complejidad:
  - "Fácil" → Aplicación directa de conceptos.
  - "Intermedio / Difícil" → Análisis, integración, hipótesis y decisión bajo incertidumbre.

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
      "contexto": "Extracto o fragmento relevante del caso en el que se basa este reto. Debe ser suficientemente completo para que otro evaluador pueda corregir la respuesta del estudiante sin leer el caso completo.",
      "pregunta": "Pregunta abierta e integradora que relacione los criterios y temas cubiertos, nivel {complejidad}",
      "respuesta_modelo": "Respuesta clara, estructurada y completa que sirva como guía de evaluación"
    }}
  ]
}}
"""