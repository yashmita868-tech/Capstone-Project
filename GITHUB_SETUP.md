# Push to GitHub & Share with Team

## One-time setup (do this on your machine)

```bash
# 1. Create a new repo on github.com, then:
cd /Users/yogeshprasad.bhondek/SNOWVIBE/GAIPT/capstone

git init
git add .
git commit -m "Initial commit — Doc Agent skeleton"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## What teammates do (in Google Colab)

1. Open `colab_team_launcher.ipynb` → **Open in Colab** button (or share the file directly)
2. Set `GITHUB_REPO` in Cell 1 to your repo URL
3. Add `ANTHROPIC_API_KEY` to Colab Secrets (🔑 left sidebar)
4. **Runtime → Run all**
5. Open the Cloudflare URL printed by Cell 5

## What's in each notebook

| File | Purpose |
|------|---------|
| `colab_team_launcher.ipynb` | **Team use** — clones GitHub, runs Streamlit app via public URL |
| `invoice_agent_colab.ipynb` | **Standalone use** — all code inlined, no GitHub needed, cell-by-cell pipeline |

## Protecting secrets

`.gitignore` should contain:
```
.env
.venv/
data/invoices.db
data/chroma/
__pycache__/
*.pyc
```
