"""
Prompt para generar rutas de retos formativos en casos clínicos de psicología
"""

RUTA_CLINICO_PSICOLOGIA_PROMPT = """
## Objetivo
Genera una **ruta de cinco retos clínicos de psicología** derivados directamente del **caso clínico psicológico proporcionado**.
Cada reto debe representar una **etapa secuencial del proceso clínico** (evaluación, diagnóstico, formulación, intervención y seguimiento) y debe mantener **coherencia total** con la información narrativa, diagnóstica y terapéutica del caso.

---

## Datos curriculares
- Competencia: {competencia}
- Capacidad: {capacidad}
- Criterio: {criterio}
- Complejidad: {complejidad}

### Temas clave
{temas_formateados}

### Caso base
{caso}

---

## Reglas generales
1. **Usa únicamente la información del caso clínico** (síntomas, historia, resultados de pruebas, hipótesis, contexto). No inventes ni añadas datos nuevos.
2. Todas las **preguntas** deben referirse a situaciones clínicas reales descritas en el caso.
3. Las **respuestas modelo** deben:
   - Ser analíticas, técnicas y pedagógicamente claras.
   - Mantener consistencia temporal y conceptual con el caso.
   - Estar alineadas con la evidencia científica (DSM-5-TR, ICD-11, guías APA o NICE).
4. Los **conceptos clave** deben incluir primero el **tema o proceso clínico central** del reto (el mismo de la subpregunta), seguido de otros conceptos o técnicas relevantes, separados por comas y terminados en punto.
5. **La complejidad ({complejidad})** debe reflejarse en:
   - El nivel de análisis requerido en las preguntas (desde identificación básica hasta razonamiento clínico avanzado).
   - La profundidad de la respuesta modelo (más detallada, integradora y crítica conforme aumenta la complejidad).
   - La integración de conceptos teóricos y aplicados.
6. Respeta exactamente la siguiente **estructura de salida** para cada uno de los cinco retos.

---

@Nombre: [Título general de la ruta.]

### **Reto 1 - Evaluación clínica adecuada**
@Reto: [Título breve del desafío]
@Contexto: [Situación inicial del caso donde se requiere decidir qué evaluación o pruebas aplicar. Menciona síntomas o comportamientos observados.]
@Pregunta: [Pregunta principal sobre selección o justificación de evaluación + subpregunta que aplique el tema clave **validez diagnóstica, evaluación multimétodo, consistencia interna, correlato clínico o sesgo de respuesta.**]
@Respuesta Modelo: [Respuesta coherente que proponga batería de evaluación apropiada al caso y justifique los instrumentos.]
@Conceptos Clave: [Tema clave exacto de la subpregunta, seguido de otros conceptos clínicos relacionados.]

---

### **Reto 2 - Diagnóstico clínico justificado**
@Reto: [Título breve del desafío]
@Contexto: [Fragmento del caso donde los síntomas y resultados apuntan a posibles diagnósticos.]
@Pregunta: [Pregunta principal sobre el diagnóstico + subpregunta que relacione con el tema **criterios diagnósticos, diagnóstico diferencial, comorbilidad, curso clínico o significancia funcional.**]
@Respuesta Modelo: [Respuesta que justifique un diagnóstico según DSM-5-TR o ICD-11, articulando datos clínicos y razonamiento.]
@Conceptos Clave: [Tema clave de la subpregunta, seguido de otros conceptos aplicables.]

---

### **Reto 3 - Formulación del caso (modelo explicativo del problema)**
@Reto: [Título breve del desafío]
@Contexto: [Momento del caso donde debe integrarse información para explicar el origen y mantenimiento del problema.]
@Pregunta: [Pregunta principal sobre la explicación del problema + subpregunta relacionada con **conceptualización del caso, esquema cognitivo, factores predisponentes, precipitantes, mantenedores o protectores.**]
@Respuesta Modelo: [Formulación coherente con el modelo teórico del caso (cognitivo, psicodinámico, sistémico, etc.) que relacione pensamientos, emociones, conductas y contexto.]
@Conceptos Clave: [Tema clave exacto, seguido de conceptos o procesos relacionados.]

---

### **Reto 4 - Plan de intervención terapéutica basado en evidencia**
@Reto: [Título breve del desafío]  
@Contexto: [Fragmento del caso donde se define la necesidad de intervención. Incluye objetivos o obstáculos clínicos relevantes.]  
@Pregunta: [Pregunta principal sobre el plan de tratamiento + subpregunta que use el tema **tratamiento basado en evidencia, objetivos SMART, exposición, reestructuración cognitiva o entrenamiento en relajación.**]  
@Respuesta Modelo: [Propuesta estructurada de intervención (fases, técnicas y objetivos) alineada con la evidencia científica y el enfoque clínico.]  
@Conceptos Clave: [Tema clave exacto de la subpregunta seguido de otros conceptos terapéuticos.]

---

### **Reto 5 - Evaluación y seguimiento**
@Reto: [Título breve del desafío]  
@Contexto: [Etapa final del caso donde se evalúan resultados, adherencia o posibles ajustes. Menciona decisiones o plazos.]  
@Pregunta: [Pregunta principal sobre evaluación de progreso + subpregunta que use **seguimiento clínico, cambio terapéutico, adherencia, revisión de hipótesis o mejora sostenida.**]  
@Respuesta Modelo: [Reflexión técnica sobre indicadores de progreso, ajustes posibles y aprendizajes clínicos, coherente con la evolución del caso.]  
@Conceptos Clave: [Tema clave exacto de la subpregunta seguido de otros conceptos relacionados.]

---

### Consideraciones finales
- Redacta en **formato narrativo profesional, claro y didáctico**.  
- Cada reto debe derivarse **directamente del caso clínico** generado.  
- Mantén **coherencia progresiva**: los retos deben leerse como una ruta clínica integrada.
"""