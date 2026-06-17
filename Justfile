artifact_repo := env_var_or_default("ROUTER_ARTIFACT_REPO", "micdn/llm-router-goose-public")
port := env_var_or_default("PORT", "4000")

default:
    just --list

# Download active public router checkpoints from Hugging Face.
artifacts:
    @.venv/bin/python scripts/router_artifacts.py download --repo "{{artifact_repo}}"

# Print Goose commands for using the router.
goose-instructions:
    @.venv/bin/python scripts/router_artifacts.py goose-instructions --repo "{{artifact_repo}}" --port "{{port}}"

# Download artifacts if needed, print Goose instructions, then run the router.
run-router:
    @ROUTER_ARTIFACT_REPO="{{artifact_repo}}" PORT="{{port}}" ./scripts/run-public-router.sh

# Restart the local router using the active combined pool.
restart-router:
    @PORT="{{port}}" ./scripts/restart-router.sh

# Upload the active public artifact bundle to Hugging Face.
upload-artifacts:
    @.venv/bin/python scripts/router_artifacts.py upload --repo "{{artifact_repo}}"
