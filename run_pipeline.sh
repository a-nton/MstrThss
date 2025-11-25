#!/bin/bash

echo "=========================================="
echo "      GDELT PROJECT AUTO-LAUNCHER"
echo "=========================================="

# 0. Set Context to Script Directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || { echo "Failed to enter script directory"; exit 1; }

echo "[INFO] Working directory set to: $SCRIPT_DIR"

# 1. Check VENV Creation
if [ ! -d ".venv" ]; then
    echo "[INFO] Virtual environment not found. Creating one..."
    python3 -m venv .venv
else
    echo "[INFO] Using existing virtual environment."
fi

# 2. Activate & Install Requirements (ALWAYS RUNS)
source .venv/bin/activate

echo "[INFO] Checking/Installing requirements..."
.venv/bin/pip install -r requirements.txt

# 3. Menu
echo ""
echo "Select an option:"
echo "1. Run Pipeline with Plugin Selection (Interactive)"
echo "2. Git: View status & commit changes"
echo "3. Exit"
read -p "Enter choice: " choice

if [ "$choice" == "1" ]; then
    echo "Launching pipeline..."
    .venv/bin/python launcher.py
elif [ "$choice" == "2" ]; then
    while true; do
        echo ""
        echo "========== GIT MENU =========="
        echo "1. View status"
        echo "2. View commit history"
        echo "3. Add all untracked files"
        echo "4. Commit all changes"
        echo "5. Create new branch"
        echo "6. Switch branch"
        echo "7. Revert to earlier commit"
        echo "8. View diff (uncommitted changes)"
        echo "9. Back to main menu"
        read -p "Git option: " git_choice

        if [ "$git_choice" == "1" ]; then
            echo ""
            git status

        elif [ "$git_choice" == "2" ]; then
            echo ""
            echo "========== COMMIT HISTORY =========="
            git log --oneline -15 2>/dev/null || echo "No commits yet"

        elif [ "$git_choice" == "3" ]; then
            echo ""
            echo "========== UNTRACKED FILES =========="
            untracked=$(git ls-files --others --exclude-standard)
            if [ -z "$untracked" ]; then
                echo "No untracked files"
            else
                echo "$untracked"
                echo ""
                read -p "Add all these files? (yes/no): " confirm
                if [ "$confirm" == "yes" ]; then
                    git add -A
                    echo "✓ All untracked files added to staging"
                    git status --short
                else
                    echo "Cancelled"
                fi
            fi

        elif [ "$git_choice" == "4" ]; then
            git status --short
            read -p "Enter commit message: " commit_msg
            git add -A
            git commit -m "$commit_msg"
            echo "Committed!"

        elif [ "$git_choice" == "5" ]; then
            echo "Current branches:"
            git branch
            read -p "New branch name: " branch_name
            git checkout -b "$branch_name"
            echo "Created and switched to '$branch_name'"

        elif [ "$git_choice" == "6" ]; then
            echo "Available branches:"
            git branch -a
            read -p "Branch to switch to: " branch_name
            git checkout "$branch_name"

        elif [ "$git_choice" == "7" ]; then
            echo ""
            echo "========== RECENT COMMITS =========="
            git log --oneline -10
            echo ""
            echo "WARNING: This will discard uncommitted changes!"
            read -p "Enter commit hash to revert to (or 'cancel'): " commit_hash
            if [ "$commit_hash" != "cancel" ]; then
                read -p "Are you sure? This cannot be undone. (yes/no): " confirm
                if [ "$confirm" == "yes" ]; then
                    git checkout -- .
                    git reset --hard "$commit_hash"
                    echo "Reverted to $commit_hash"
                fi
            fi

        elif [ "$git_choice" == "8" ]; then
            echo ""
            git diff

        elif [ "$git_choice" == "9" ]; then
            break
        fi
    done
elif [ "$choice" == "3" ]; then
    echo "Goodbye!"
fi

echo "Done."