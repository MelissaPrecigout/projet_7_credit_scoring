#!/bin/bash
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0:${PORT:-8000} api:app --timeout 600