# macOS Terminal Setup Guide

This guide describes a practical macOS terminal setup based on iTerm2 and Oh My Zsh. It keeps the existing Oh My Zsh installation, adds a modern prompt, improves navigation and search, and introduces a small set of reliable command-line tools.

The setup is designed for daily development work: fast directory navigation, readable file listings, better history search, Git-aware prompts, syntax highlighting, and a maintainable `.zshrc`.

## Goals

- Keep iTerm2 as the terminal application.
- Keep Oh My Zsh for Zsh framework features and plugin loading.
- Use Starship for the prompt instead of an Oh My Zsh theme.
- Add a small number of high-value Zsh plugins.
- Add modern replacements for common Unix commands.
- Keep configuration readable and easy to debug.

## Prerequisites

You already have:

- macOS
- iTerm2
- Oh My Zsh

Recommended package manager:

- Homebrew

Check Homebrew:

```zsh
brew --version
```

If Homebrew is missing, install it from:

```text
https://brew.sh
```

## Install Core Tools

Install the recommended tools:

```zsh
brew install starship
brew install zoxide
brew install fzf
brew install eza
brew install bat
brew install fd
brew install ripgrep
brew install jq
brew install tree
brew install wget
brew install neovim
```

Install a Nerd Font for icons:

```zsh
brew install --cask font-meslo-lg-nerd-font
```

## iTerm2 Configuration

Open iTerm2 settings:

```text
iTerm2 > Settings > Profiles
```

Recommended settings:

- `Text > Font`: `MesloLGS Nerd Font`
- `Text > Font size`: `14` or `15`
- `Terminal > Scrollback lines`: `100000`
- `Terminal > Silence bell`: enabled
- `Keys > Left Option Key`: `Esc+`
- `Keys > Right Option Key`: `Esc+`

The `Esc+` option makes shortcuts such as `Alt+C` work correctly with `fzf`.

Useful iTerm2 shortcuts:

| Shortcut | Action |
| --- | --- |
| `Cmd+T` | New tab |
| `Cmd+D` | Split pane vertically |
| `Cmd+Shift+D` | Split pane horizontally |
| `Cmd+Option+Arrow` | Move between panes |
| `Cmd+K` | Clear screen |

## Tool Overview

### Starship

Starship is a fast, cross-shell prompt. It shows context such as the current directory, Git branch, Git status, language versions, command duration, and exit status.

Why use it:

- Faster and more consistent than many large Zsh themes.
- Works across Zsh, Bash, Fish, Nushell, PowerShell, and more.
- Config lives in `~/.config/starship.toml`.

Install:

```zsh
brew install starship
```

Create an initial config:

```zsh
mkdir -p ~/.config
starship preset nerd-font-symbols -o ~/.config/starship.toml
```

Enable in `~/.zshrc`:

```zsh
eval "$(starship init zsh)"
```

Useful commands:

```zsh
starship preset --list
starship explain
starship timings
```

Common customizations in `~/.config/starship.toml`:

```toml
add_newline = false

[directory]
truncation_length = 4
truncate_to_repo = true

[cmd_duration]
min_time = 1000

[git_status]
disabled = false
```

### zoxide

`zoxide` is a smarter `cd`. It remembers directories you use and lets you jump to them by partial name.

Why use it:

- Avoids typing long paths repeatedly.
- Learns from your actual navigation habits.
- Works especially well across project directories.

Install:

```zsh
brew install zoxide
```

Enable in `~/.zshrc`:

```zsh
eval "$(zoxide init zsh)"
```

Usage:

```zsh
cd ~/Projects/stochaflow
z stochaflow
z projects
zi
```

Commands:

| Command | Purpose |
| --- | --- |
| `z name` | Jump to a remembered directory matching `name` |
| `zi` | Interactive directory jump |
| `zoxide query name` | Show the path that would be selected |
| `zoxide remove path` | Remove a path from the database |

Recommendation: use `z` for a while before aliasing over `cd`. Keeping `cd` unchanged is easier to debug.

### fzf

`fzf` is a fuzzy finder for interactive search. It is useful for command history, files, directories, Git branches, and custom shell workflows.

Why use it:

- Fast interactive filtering.
- Great for searching long shell history.
- Integrates with Zsh keybindings.

Install:

```zsh
brew install fzf
```

Enable in `~/.zshrc`:

```zsh
source <(fzf --zsh)
```

Common shortcuts:

| Shortcut | Purpose |
| --- | --- |
| `Ctrl+R` | Search command history |
| `Ctrl+T` | Search files and paste selected path |
| `Alt+C` | Search directories and `cd` into one |

Useful commands:

```zsh
fzf
history | fzf
git branch | fzf
```

Optional preview setup:

```zsh
export FZF_DEFAULT_OPTS="--height 40% --layout=reverse --border"
export FZF_CTRL_T_OPTS="--preview 'bat --style=numbers --color=always --line-range :200 {}'"
export FZF_ALT_C_OPTS="--preview 'eza --tree --level=2 --icons {}'"
```

### eza

`eza` is a modern replacement for `ls`. It supports icons, Git status, tree view, colors, and better defaults.

Why use it:

- More readable file listings.
- Git-aware output.
- Tree views without needing separate tools for many cases.

Install:

```zsh
brew install eza
```

Usage:

```zsh
eza
eza -lh
eza -lah
eza --tree --level=2
eza -lh --git --icons
```

Recommended aliases:

```zsh
alias ls="eza --icons"
alias ll="eza -lh --icons --git"
alias la="eza -lah --icons --git"
alias tree="eza --tree --icons"
```

### bat

`bat` is a modern replacement for `cat`. It adds syntax highlighting, line numbers, Git change markers, and paging.

Why use it:

- Much easier to read source files in the terminal.
- Excellent as an `fzf` preview command.
- Handles many languages automatically.

Install:

```zsh
brew install bat
```

Usage:

```zsh
bat README.md
bat -n src/main.py
bat --style=numbers --line-range 1:120 file.py
```

Recommended alias:

```zsh
alias cat="bat"
```

If you sometimes need plain `cat`, call it explicitly:

```zsh
command cat file.txt
```

### fd

`fd` is a user-friendly replacement for `find`. It is fast, respects `.gitignore` by default, and has simpler syntax.

Why use it:

- Easier than `find` for common searches.
- Very fast in project directories.
- Works well with `fzf`.

Install:

```zsh
brew install fd
```

Usage:

```zsh
fd README
fd "\.py$"
fd test
fd -H env
fd -e md
```

Examples:

```zsh
fd -e py
fd -e md docs
fd --hidden --exclude .git
```

Recommended alias:

```zsh
alias find="fd"
```

If a script expects POSIX `find`, use:

```zsh
command find . -name "*.py"
```

### ripgrep

`ripgrep`, usually called `rg`, is a fast replacement for `grep`. It recursively searches text and respects `.gitignore` by default.

Why use it:

- Extremely fast source search.
- Better defaults for codebases.
- Clear output with file names and line numbers.

Install:

```zsh
brew install ripgrep
```

Usage:

```zsh
rg "TODO"
rg "class App"
rg -n "pattern"
rg -i "pattern"
rg --hidden "pattern"
rg --glob "*.py" "pattern"
```

Recommended alias:

```zsh
alias grep="rg"
```

Use system `grep` explicitly when needed:

```zsh
command grep -R "pattern" .
```

### jq

`jq` is a command-line JSON processor. It formats, filters, and transforms JSON.

Why use it:

- Makes API output readable.
- Useful for scripts and debugging.
- Works well with `curl`.

Install:

```zsh
brew install jq
```

Usage:

```zsh
echo '{"name":"stochaflow"}' | jq
curl -s https://api.github.com/repos/sharkdp/bat | jq '.stargazers_count'
jq '.scripts' package.json
```

Common filters:

```zsh
jq '.name'
jq '.items[] | .name'
jq -r '.path'
```

### neovim

Neovim is a modern Vim-based editor. In this setup it is used as the default terminal editor.

Why use it:

- Fast terminal editing.
- Good Git and CLI integration.
- Can grow into a full IDE, but does not have to.

Install:

```zsh
brew install neovim
```

Set as default editor:

```zsh
export EDITOR="nvim"
export VISUAL="nvim"
```

Usage:

```zsh
nvim ~/.zshrc
nvim ~/.config/starship.toml
```

### wget

`wget` downloads files from URLs. macOS includes `curl`, but many Linux guides and scripts use `wget`.

Install:

```zsh
brew install wget
```

Usage:

```zsh
wget https://example.com/file.zip
wget -O output.zip https://example.com/file.zip
```

### tree

`tree` displays directory structures. `eza --tree` covers many cases, but `tree` is still useful because many guides and scripts expect it.

Install:

```zsh
brew install tree
```

Usage:

```zsh
tree
tree -L 2
tree -a -I ".git|node_modules"
```

## Oh My Zsh Plugins

Keep the plugin list short. Too many plugins slow down shell startup and make debugging harder.

Install extra plugins:

```zsh
git clone https://github.com/zsh-users/zsh-autosuggestions \
  ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

git clone https://github.com/zsh-users/zsh-syntax-highlighting.git \
  ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

git clone https://github.com/zsh-users/zsh-history-substring-search \
  ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search
```

Recommended plugin list in `~/.zshrc`:

```zsh
plugins=(
  git
  brew
  macos
  zoxide
  fzf
  zsh-autosuggestions
  zsh-history-substring-search
  zsh-syntax-highlighting
)
```

Plugin descriptions:

| Plugin | Purpose |
| --- | --- |
| `git` | Adds Git aliases and helper functions |
| `brew` | Adds Homebrew completions and helpers |
| `macos` | Adds macOS utility aliases/functions |
| `zoxide` | Integrates `zoxide` with Zsh |
| `fzf` | Integrates `fzf` keybindings and completion |
| `zsh-autosuggestions` | Shows suggestions from history as you type |
| `zsh-history-substring-search` | Search history by typed substring |
| `zsh-syntax-highlighting` | Highlights valid/invalid commands while typing |

Important: keep `zsh-syntax-highlighting` last in the plugin list.

## Recommended `.zshrc`

Back up your current config before editing:

```zsh
cp ~/.zshrc ~/.zshrc.backup.$(date +%Y%m%d-%H%M%S)
```

Recommended structure:

```zsh
export ZSH="$HOME/.oh-my-zsh"

# Starship will handle the prompt.
ZSH_THEME=""

plugins=(
  git
  brew
  macos
  zoxide
  fzf
  zsh-autosuggestions
  zsh-history-substring-search
  zsh-syntax-highlighting
)

source "$ZSH/oh-my-zsh.sh"

# XDG-style paths
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_STATE_HOME="$HOME/.local/state"

# Editor
export EDITOR="nvim"
export VISUAL="nvim"

# PATH
export PATH="$HOME/.local/bin:$PATH"
export PATH="/opt/homebrew/bin:$PATH"

# History
HISTFILE="$HOME/.zsh_history"
HISTSIZE=100000
SAVEHIST=100000

setopt APPEND_HISTORY
setopt SHARE_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE
setopt HIST_EXPIRE_DUPS_FIRST
setopt HIST_FIND_NO_DUPS
setopt AUTOCD
setopt NO_BEEP
setopt NUMERIC_GLOB_SORT

# Modern CLI aliases
alias ls="eza --icons"
alias ll="eza -lh --icons --git"
alias la="eza -lah --icons --git"
alias tree="eza --tree --icons"
alias cat="bat"
alias grep="rg"
alias find="fd"

# Git aliases
alias gs="git status"
alias ga="git add"
alias gc="git commit"
alias gp="git push"
alias gl="git log --oneline --graph --decorate"
alias gd="git diff"

# fzf
source <(fzf --zsh)
export FZF_DEFAULT_OPTS="--height 40% --layout=reverse --border"
export FZF_CTRL_T_OPTS="--preview 'bat --style=numbers --color=always --line-range :200 {}'"
export FZF_ALT_C_OPTS="--preview 'eza --tree --level=2 --icons {}'"

# zoxide
eval "$(zoxide init zsh)"

# Starship prompt
eval "$(starship init zsh)"
```

Reload:

```zsh
source ~/.zshrc
```

## Optional File Split

If `.zshrc` becomes large, split your config:

```text
~/.config/zsh/
  aliases.zsh
  exports.zsh
  fzf.zsh
  git.zsh
```

Then load them from `~/.zshrc`:

```zsh
for file in "$HOME"/.config/zsh/*.zsh; do
  source "$file"
done
```

This keeps `~/.zshrc` short while preserving Oh My Zsh compatibility.

## Recommended Starship Config

Create or edit:

```zsh
nvim ~/.config/starship.toml
```

Suggested config:

```toml
add_newline = false
command_timeout = 1000

[directory]
truncation_length = 4
truncate_to_repo = true
read_only = " ro"

[git_branch]
symbol = "git "

[git_status]
disabled = false
ahead = "ahead ${count}"
behind = "behind ${count}"
diverged = "ahead ${ahead_count} behind ${behind_count}"
modified = "modified"
staged = "staged"
untracked = "untracked"

[cmd_duration]
min_time = 1000
format = "took [$duration]($style) "

[character]
success_symbol = "[>](bold green)"
error_symbol = "[>](bold red)"
vimcmd_symbol = "[<](bold green)"
```

This config avoids heavy decoration while keeping useful status visible.

## Verification Checklist

Open a new iTerm2 tab and run:

```zsh
echo $SHELL
starship --version
zoxide --version
fzf --version
eza --version
bat --version
fd --version
rg --version
jq --version
nvim --version
```

Check interactive behavior:

- Type part of an old command and confirm autosuggestions appear.
- Type an invalid command and confirm syntax highlighting changes color.
- Press `Ctrl+R` and confirm history search opens.
- Press `Alt+C` and confirm directory search opens.
- Run `ll` in a Git repository and confirm file listing includes icons and Git status.
- Enter a Git repository and confirm the prompt shows branch information.

## Troubleshooting

### Icons display as boxes

Set iTerm2 font to a Nerd Font:

```text
iTerm2 > Settings > Profiles > Text > Font > MesloLGS Nerd Font
```

### `Alt+C` does not open fzf directory search

Set Option key behavior:

```text
iTerm2 > Settings > Profiles > Keys > Left Option Key > Esc+
```

### Shell startup is slow

Measure startup time:

```zsh
time zsh -i -c exit
```

Common causes:

- Too many Oh My Zsh plugins.
- Expensive commands running in `.zshrc`.
- Network calls during shell startup.
- Slow prompt modules.

Use:

```zsh
starship timings
```

### `cat`, `find`, or `grep` behaves differently

The aliases point to modern replacements:

```zsh
cat -> bat
find -> fd
grep -> rg
```

Use the original command with `command`:

```zsh
command cat file.txt
command find . -name "*.py"
command grep -R "pattern" .
```

### Homebrew path is wrong

Apple Silicon usually uses:

```zsh
/opt/homebrew/bin
```

Intel Macs usually use:

```zsh
/usr/local/bin
```

You can make this portable:

```zsh
if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
  eval "$(/usr/local/bin/brew shellenv)"
fi
```

## Maintenance

Update Homebrew tools:

```zsh
brew update
brew upgrade
```

Update Oh My Zsh:

```zsh
omz update
```

Update custom plugins:

```zsh
cd ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && git pull --ff-only
cd ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && git pull --ff-only
cd ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search && git pull --ff-only
```

Review `.zshrc` periodically. If a tool or alias is unused, remove it. A terminal setup should stay boring, fast, and easy to repair.
