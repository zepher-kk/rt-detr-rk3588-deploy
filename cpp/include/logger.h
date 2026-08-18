#pragma once

// ============================================================================
// logger.h - 等级化模块化日志系统
//
// 用法（流式，保持原有打印风格）：
//   LOG(MOD_MAIN, LOG_ERROR) << "Failed to load image: " << path << "\n";
//   LOGT(MOD_PIPELINE, LOG_ERROR, "Pre") << "no source\n";   // 自定义子标签
//
// -G/--DEBUG 模块级联控制（main 解析后调用 log_set_modules）：
//   -G 0 : 仅打印 ERROR（任何模块恒打印，错误不可隐藏）
//   -G N : 开启前 N 个模块：1=[Main] 2=[Main]+[RKNN] 3=+[Pipeline] ... 8=全部
//   默认（不传 -G）：全部模块开启，级别 INFO
//
// 级别：ERROR < WARN < INFO < DEBUG < TRACE
// ============================================================================

#include <iostream>

// 模块枚举（顺序即 -G 级联顺序，与 -G 说明一致）
enum LogModule
{
	MOD_MAIN = 0,
	MOD_RKNN,
	MOD_PIPELINE,
	MOD_RGA,
	MOD_DRM,
	MOD_V4L2,
	MOD_POST,
	MOD_TEST,
	MOD_COUNT
};

// 日志级别
enum LogLevel
{
	LOG_ERROR = 0,
	LOG_WARN  = 1,
	LOG_INFO  = 2,
	LOG_DEBUG = 3,
	LOG_TRACE = 4
};

// 模块显示标签
inline const char* module_tag(int module)
{
	static const char* tags[MOD_COUNT] =
	{
		"Main", "RKNN", "Pipeline", "RGA", "DRM", "V4L2", "Post", "Test"
	};
	return (module >= 0 && module < MOD_COUNT) ? tags[module] : "?";
}

// 全局状态（默认：全部模块开启 + INFO 级别，保持改动前行为）
inline int      g_module_mask = (1 << MOD_COUNT) - 1;
inline LogLevel g_level       = LOG_INFO;

inline void     log_set_modules(int mask) { g_module_mask = mask; }
inline int      log_modules()             { return g_module_mask; }
inline void     log_set_level(LogLevel l) { g_level = l; }
inline LogLevel log_level()               { return g_level; }

// 是否输出：ERROR 恒输出；其余需模块在掩码内且级别不高于当前级别
inline bool log_enabled(int module, LogLevel level)
{
	if (level == LOG_ERROR) return true;
	if (module < 0 || module >= MOD_COUNT) return false;
	if (!(g_module_mask & (1 << module))) return false;
	return level <= g_level;
}

// 流式日志代理：构造时判定开关并输出 [Tag] 前缀，析构时刷新
class LogStream
{
	public:
		LogStream(int module, LogLevel level, const char* tag = nullptr, bool force = false)
			: enabled_(force ? true : log_enabled(module, level))
		{
			if (enabled_)
			{
				std::cerr << "[" << (tag ? tag : module_tag(module)) << "]";
				if (level != LOG_INFO)
				{
					std::cerr << "[" << level_name(level) << "]";
				}
				std::cerr << " ";
			}
		}

		template <typename T>
		LogStream& operator<<(const T& v)
		{
			if (enabled_) std::cerr << v;
			return *this;
		}

		// 流操纵符（std::endl / std::flush / std::ends）
		LogStream& operator<<(std::ostream& (*manip)(std::ostream&))
		{
			if (enabled_) std::cerr << manip;
			return *this;
		}

		~LogStream()
		{
			if (enabled_) std::cerr.flush();
		}

	private:
		static const char* level_name(LogLevel l)
		{
			switch (l)
			{
				case LOG_ERROR: return "ERROR";
				case LOG_WARN:  return "WARN";
				case LOG_DEBUG: return "DEBUG";
				case LOG_TRACE: return "TRACE";
				default:        return "INFO";
			}
		}

		bool enabled_;
};

// 便捷宏（流式）：LOG(模块, 级别) << ... / LOGT(模块, 级别, 子标签) << ...
#define LOG(module, level)        LogStream(module, level)
#define LOGT(module, level, tag)  LogStream(module, level, tag)
#define LOGR(module)              LogStream(module, LOG_INFO, nullptr, true)
