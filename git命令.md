### 克隆仓库

地址使用 SSH 形式（从 GitHub 仓库页面可以选择 SSH 并复制）：

git clone git@github.com:你的用户名/learn-git.git

替换“你的用户名”和仓库名。执行后，会多出一个 learn-git（仓库名） 文件夹，这就是本地仓库。进入它：cd learn-git



### 创建分支

git switch -c first-feature   # 创建并切换到 first-feature 分支

可以随时用 git branch 查看所有本地分支，当前所在分支前会有 \*





### 在本地做修改并提交 —— git add 和 git commit

查看状态：git status   会显示 “Untracked files: hello.txt”，意思是该文件还没有被 Git 跟踪。

添加改动到暂存区：

&#x09;			git add hello.txt

&#x09;			# 或一次性添加所有改动

&#x09;			git add .

再运行 git status，会看到文件变成绿色，处于 “Changes to be committed” 状态。

提交到本地仓库：git commit -m "Add hello.txt with greeting message"

-m 后跟提交说明，描述你做了什么。

这条提交现在只存在于你的本地电脑上，还没到 GitHub。



### 推送到 GitHub —— git push

把当前分支推送到远程仓库：git push -u origin main

origin 是远程仓库的默认名称（克隆时自动设置）。

-u 会将本地分支与远程分支关联，以后再推送只需 git push 即可。

刷新 GitHub 上的仓库页面，你会看到一个黄色的提示框，告知有新分支 first-feature 被推送。你也可以切换到该分支查看文件。

