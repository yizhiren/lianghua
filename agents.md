# 发布规则

每次修改代码或配置后，必须严格按以下顺序执行：

1. 先运行与改动相关的自测；自测失败时不得继续发布。
2. 自测通过后提交 Git commit。
3. 将提交 push 到 GitHub 远程仓库 `git@github.com:yizhiren/lianghua.git`。
4. 登录部署机 `qroot@192.168.3.176`，拉取最新代码。
5. 使用 Docker Compose 重新构建并部署服务。
6. 部署后验证容器状态、前端访问和 API 健康检查。

不得跳过自测、commit、push 或部署后的验证；密码、API key 等敏感信息不得写入仓库。
