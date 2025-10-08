"""
Prompt para generar rutas de retos formativos en el método ABP (Aprendizaje Basado en Proyectos).
"""

RUTA_ABP_PROMPT = """
## Objetivo
Generar **cinco retos formativos secuenciales** alineados con las etapas del **Aprendizaje Basado en Proyectos (ABP)**, utilizando el caso o proyecto proporcionado y los datos curriculares.
Cada reto debe evaluar una habilidad o etapa del proceso ABP, manteniendo la coherencia con los temas clave indicados y respetando la estructura obligatoria.

---

## Formato obligatorio
El modelo debe usar exactamente el siguiente formato:

@Nombre: [Título general de la ruta.]

(Después, por cada reto:)

@Reto: [Título breve del desafío]
@Contexto: [Describa el contexto del reto usando hechos, actores o tensiones del proyecto.]
@Pregunta: [Incluya **2 o 3 preguntas clave**, formuladas de manera secuencial según la etapa correspondiente. La **última subpregunta** debe aplicar directamente **uno de los temas clave** listados en {temas_formateados}.]
@Respuesta Modelo: [Responda cada pregunta de forma analítica, contextual y coherente, siguiendo el mismo orden en que fueron planteadas.]
@Conceptos Clave: [**Debe iniciar con el mismo tema clave exacto usado en la subpregunta final**, seguido de otros conceptos o herramientas relacionados. Separe por comas y termine con punto.]

---

## Datos curriculares
- Competencia: {competencia}
- Capacidad: {capacidad}
- Criterio: {criterio}
- Complejidad: {complejidad}

### Temas clave
{temas_formateados}

### Proyecto base
{caso}

## Etapas del ABP y retos a generar

1. **Análisis del problema y contexto**
   - Propósito: Comprender la situación inicial y sus causas.
   - El reto debe centrarse en identificar y explicar el problema base del proyecto.

2. **Identificación de necesidades, requisitos y objetivos**
   - Propósito: Formular objetivos y definir criterios de éxito basados en las necesidades detectadas.
   - El reto debe requerir inferencia o análisis de información contextual.

3. **Diseño y propuesta de solución**
   - Propósito: Crear una propuesta viable, coherente y justificada frente al problema.
   - El reto debe conectar la solución con los temas clave y los objetivos definidos.

4. **Evaluación de riesgos y limitaciones**
   - Propósito: Identificar factores críticos que podrían obstaculizar la implementación del proyecto.
   - El reto debe promover pensamiento crítico y toma de decisiones.

5. **Validación y mejora del producto final**
   - Propósito: Reflexionar sobre la ejecución, aprendizajes, resultados y mejoras posibles.
   - El reto debe incluir elementos de cierre: decisiones tomadas, impacto, sostenibilidad.

---

## Instrucciones específicas

1. Inicie con un único `@Nombre` general que represente toda la ruta.
2. Genere **cinco retos**, uno por cada etapa del ABP, respetando la estructura obligatoria.
3. Cada reto debe contener **2 o 3 preguntas** y sus respuestas modelo correspondientes.
4. Alinee cada reto con los objetivos y temas clave definidos.
5. **Nivel de complejidad:**
   - “Fácil”: Aplicación directa de conceptos o métodos.
   - “Intermedio o Difícil”: Requiere análisis, integración, justificación o toma de decisiones.
6. Evite repetir ideas o contextos entre retos.
7. Use lenguaje técnico, académico y preciso.
8. Devuelva los cinco retos juntos, sin explicaciones adicionales ni etiquetas nuevas.
9. Verifique que cada reto tenga **exactamente** estas secciones:
`@Reto`, `@Contexto`, `@Pregunta`, `@Respuesta Modelo`, `@Conceptos Clave`.
10. **Trazabilidad obligatoria:** 
    - La **última subpregunta** debe aplicar directamente **uno de los temas clave** listados en {temas_formateados}.  
    - Ese mismo tema debe aparecer **primero en `@Conceptos Clave`**, escrito exactamente igual.
"""