RUTA_ABP_PROMPT = """
### Instrucción
Eres un experto en diseño pedagógico. Genera una ruta de aprendizaje con exactamente 5 retos,
uno por cada etapa del Aprendizaje Basado en Proyectos (ABP).
Cada reto debe integrar y relacionar múltiples criterios y campos temáticos de forma cohesionada.

### Estructura pedagógica del curso:
{competencias_formateadas}

### Proyecto base:
{caso}

### Nivel de complejidad: {complejidad}

### Reglas de generación:
- Genera exactamente 5 retos, uno por cada etapa ABP, en el orden indicado:
  1. Análisis del problema y contexto.
  2. Identificación de necesidades, requisitos y objetivos.
  3. Diseño y propuesta de solución.
  4. Evaluación de riesgos y limitaciones.
  5. Validación y mejora del producto final.
- Cada reto debe ser integrador: su pregunta debe exigir relacionar conceptos de varios criterios o temas.
- Distribuye la cobertura de criterios de forma que el conjunto de retos abarque todos los criterios disponibles.
- Por cada criterio seleccionado en un reto, indica ÚNICAMENTE los CampoTematicoId que ese reto
  trabaja dentro de ese criterio. No mezcles campos de distintos criterios en la misma lista.
- La pregunta de cada reto debe contener exactamente 2 o 3 subpreguntas, separadas por \n,
  sin guiones ni numeración, cada una terminada con ?
- Basa los retos en el proyecto proporcionado y en los criterios y campos temáticos de la estructura pedagógica.
- Nivel de complejidad:
  - "Fácil" → Aplicación directa de conceptos.
  - "Intermedio / Difícil" → Análisis, integración, justificación y toma de decisiones.

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
      "contexto": "Extracto o fragmento relevante del proyecto en el que se basa este reto. Debe ser suficientemente completo para que otro evaluador pueda corregir la respuesta del estudiante sin leer el proyecto completo.",
      "pregunta": "Primera subpregunta?\nSegunda subpregunta?\nTercera subpregunta?",
      "respuesta_modelo": "Respuesta analítica a cada subpregunta, en el mismo orden en que fueron planteadas."
    }}
  ]
}}
"""