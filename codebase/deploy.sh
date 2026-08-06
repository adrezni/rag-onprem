#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# Deploys the PDF Chat RAG app into the current OpenShift project.
#
# Run this from inside your JupyterLab workbench, from the codebase/
# directory (the one containing app.py, rag_engine.py, config.py,
# requirements.txt, and the Dockerfile).
#
# Safe to re-run: creates resources on first run, and on later runs
# just rebuilds the image from your current local files (including
# any edits you've made to config.py) and rolls out the update.
#
# Usage:
#   cd ~/rag-onprem/codebase
#   chmod +x deploy.sh
#   ./deploy.sh
# ─────────────────────────────────────────────────────────────────
set -e

APP_NAME="pdf-chat-app"
NAMESPACE=$(oc project -q)

echo "=================================================="
echo " Deploying '$APP_NAME' to project: $NAMESPACE"
echo "=================================================="

# ─ 1. BuildConfig + ImageStream (binary source — builds from your
#       LOCAL files, so your edited config.py is what gets packaged,
#       not whatever's on GitHub) ──────────────────────────────────
if ! oc get bc/"$APP_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "--> Creating BuildConfig (binary/local source)..."
    oc new-build --name="$APP_NAME" --binary --strategy=docker -n "$NAMESPACE"
else
    echo "--> BuildConfig already exists, reusing it."
fi

# ─ 2. Trigger a build from the current directory ──────────────────
echo "--> Building image from local files (this can take a few minutes)..."
oc start-build "$APP_NAME" --from-dir=. --follow -n "$NAMESPACE"

# ─ 3. Deployment + Service (only created once; later builds trigger
#       an automatic rollout via the ImageStream) ──────────────────
if ! oc get deployment/"$APP_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "--> Creating Deployment + Service..."
    oc new-app "$APP_NAME" --name="$APP_NAME" -n "$NAMESPACE"
else
    echo "--> Deployment already exists; new image will roll out automatically."
fi

# ─ 4. Route — explicit port avoids the 8080-vs-8501 mismatch ──────
if ! oc get route/"$APP_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "--> Exposing service on port 8501..."
    oc expose service "$APP_NAME" --port=8501 -n "$NAMESPACE"
else
    echo "--> Route already exists."
fi

echo ""
echo "=================================================="
echo " Done! Your app URL:"
echo "   http://$(oc get route "$APP_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.host}')"
echo "=================================================="