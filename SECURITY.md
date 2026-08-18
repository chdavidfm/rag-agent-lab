# Política de seguridad

## Versiones con soporte

Se da soporte a la última versión publicada en la rama `main`.

## Comunicar una vulnerabilidad

Si encuentras un fallo de seguridad, **no abras una incidencia pública**.
Escribe a **chdavidfm@gmail.com** con:

- una descripción del problema y de su impacto,
- los pasos para reproducirlo,
- la versión o el commit afectado.

Recibirás confirmación en un plazo de 72 horas y una valoración del
alcance en un máximo de 7 días.

## Notas sobre este proyecto

- Las credenciales se leen del entorno o de un archivo `.env` que **nunca**
  se versiona: está excluido en `.gitignore`.
- El modo por defecto funciona sin credenciales y sin salida a internet.
- El contenedor ejecuta el servicio con un usuario sin privilegios.
- El análisis de CodeQL se ejecuta en cada cambio y semanalmente.
