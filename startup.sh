#!/bin/bash
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0:$PORT api:app