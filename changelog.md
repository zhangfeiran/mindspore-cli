功能特性
  - 支持Ascend A2模型训练运行时失败报错、精度偏移、性能瓶颈基础分析能力
  - 支持hf transformers模型迁移，算法mhc/attn-residual集成至qwen3 skill模板
  - 集成openjiuwen claw，提供精度定位example和部署指南
  - 优化了 UI 中任务执行过程的实时反馈，在隐藏工具调用组装期间也能显示工作状态。
  - 新增编辑类工具结果的 diff 视图展示，文件改动更直观。
  - 优化了工具调用转录内容的布局与可读性。
  - 隐藏后端 agent skill 的 slash command 自动补全项，减少用户侧命令面噪音。

问题修复
  - 修复了 shell 在流式输出被截断场景下中断处理不可靠的问题。
  - 修复并统一了 bug / issue 数据结构问题。
  - 修复了工具调用转录内容的对齐与分隔问题，提升复杂交互场景下的可读性。
  - 修复了安装文档中的 curl 示例问题，避免使用 GitCode 不支持的 raw 文件地址
