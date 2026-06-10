"""
Prompt template para generar casos estratégicos breves
(Método del Caso tipo Harvard / IESE)

"""

CASO_ESTRATEGICO_PROMPT = """
## Objetivo
Redacta un **caso estratégico breve y analítico**, al estilo de **Harvard Business School** o **IESE**, destinado al análisis individual.  
No incluyas soluciones, juicios de valor ni notas didácticas.

---

## Datos curriculares
- Curso: {nombre_curso}
- Competencia: {competencia}
- Capacidad: {capacidad}
- Criterio: {criterio}
- Nivel de complejidad: {complejidad}
- Temas clave: {temas_formateados}

---

## Texto base (opcional, puede estar vacío)
{contexto}

Si el texto base está presente, extrae y reinterpreta:
- Nombre de la empresa y sector.
- Dilema o problema principal.
- Propósito o misión institucional.
- Datos relevantes (cifras, contexto, actores, tensiones).
Usa esta información para reconstruir un caso coherente y original, sin copiar texto literal.

Si no hay texto base, genera un caso nuevo coherente con los datos curriculares y el nivel de complejidad **{complejidad}**.
El dilema debe centrarse en los **temas clave** (**{temas_formateados}**) e implicar la competencia, capacidad y criterio proporcionados, sin mencionarlos directamente.

---

## Instrucciones de redacción

1. **Estilo narrativo:**
   - Profesional, claro y realista.
   - Narración objetiva, sin moralejas ni conclusiones.
   - Tono analítico y empresarial.

2. **Contenido obligatorio:**
Incluye en el cuerpo narrativo:
   - Identificación del problema estratégico.
   - Causas y tensiones principales.
   - 2-3 actores con visiones distintas y roles definidos.
   - Datos **cuantitativos y cualitativos** (cifras, conflictos, declaraciones, indicadores).
   - 2 o 3 alternativas estratégicas viables vinculadas a los temas clave, con sus:
      - Ventajas
      - Limitaciones
      - Áreas implicadas
      - Posibles efectos
   - Información operativa mínima (cargos, áreas, recursos, plazos).

---

## Estructura esperada del caso

1. **Título**  
2. **Resumen** — síntesis del contexto y dilema.  
3. **Contexto** — empresa, entorno y situación.  
4. **Datos clave** — hechos y cifras relevantes.  
5. **Problema central** — dilema o decisión pendiente.  
6. **Actores** — 2-3 personajes con perspectivas distintas.  
7. **Alternativas estratégicas** — opciones viables con pros y contras.  
8. **Información operativa mínima** — cargos, áreas, recursos, plazos.

---

## Reglas finales
- No incluir soluciones, recomendaciones ni cierre instructivo.  
- No usar listas dentro del cuerpo narrativo.  
- Prioriza coherencia, tensión estratégica y claridad.
"""