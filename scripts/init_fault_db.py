"""初始化故障码/维修记录 SQLite (Agent 工具的数据源)。

python scripts/init_fault_db.py
示例数据是演示用的, 之后可以换成你从手册里整理的真实故障码表。
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

FAULT_CODES = [
    # --- 西门子 S7-1200 PLC ---
    ("E203", "S7-1200", "CPU 内部错误: 程序块校验失败", "1.重新下载程序 2.检查存储卡 3.仍报错则恢复出厂设置"),
    ("E101", "S7-1200", "扫描周期超时(看门狗动作)", "1.检查程序死循环 2.增大最大循环时间 3.用 RE_TRIGR 指令复位循环定时器"),
    ("E305", "S7-1200", "存储卡不兼容或损坏", "1.确认为预格式化 SIMATIC 存储卡 2.换卡测试 3.检查卡槽针脚"),
    ("E410", "S7-1200", "PROFINET 通信伙伴丢失", "1.检查网线和交换机 2.核对 IP/设备名 3.查看诊断缓冲区定位断点"),
    # --- 西门子变频器 G120 ---
    ("F0001", "西门子变频器 G120", "过流故障", "1.检查电机电缆短路 2.检查加速时间是否过短 3.检查负载是否卡死"),
    ("F0002", "西门子变频器 G120", "过压故障", "1.检查电源电压 2.延长减速时间 3.加装制动电阻"),
    ("F0003", "西门子变频器 G120", "欠压故障", "1.检查进线电压 2.检查直流母线电容 3.排查电网瞬时跌落"),
    ("F0011", "西门子变频器 G120", "电机过温", "1.降低负载 2.检查电机风扇 3.核对电机额定参数设置"),
    ("F0070", "西门子变频器(USS/Modbus)", "串行链路超时: 规定时间内未收到有效数据报文", "1.检查通信电缆 2.核对波特率/站地址 3.检查 P2014 超时设置"),
    # --- 三菱 FX3U PLC ---
    ("6706", "三菱 FX3U", "指针/软元件编号超出范围", "1.检查程序中的索引寻址 2.核对软元件范围 3.重新编译下载"),
    ("6801", "三菱 FX3U", "程序语法错误", "1.用 GX Works2 程序检查 2.修正报错步序号处指令"),
    ("8060", "三菱 FX3U", "PLC 与编程口通信异常", "1.检查编程电缆 2.核对通信参数 3.检查 422 口是否损坏"),
    # --- 发那科机器人 ---
    ("E-STOP", "发那科机器人 LR Mate", "急停回路触发", "1.检查急停按钮是否复位 2.检查安全门信号 3.检查外部急停接线"),
    ("SRVO-023", "发那科机器人", "伺服过热", "1.降低负载/速度 2.检查散热风扇 3.检查环境温度是否超40°C"),
    ("SRVO-050", "发那科机器人", "碰撞检测报警", "1.确认机器人是否碰撞 2.检查负载设定是否正确 3.复位后低速试运行"),
    ("SRVO-062", "发那科机器人", "编码器电池电压低(BZAL)", "1.更换编码器电池 2.执行零点复归(Mastering) 3.确认位置数据"),
]

REPAIR_HISTORY = [
    ("2026-03-12", "S7-1200", "ERROR灯常亮, E203", "重新下载程序后恢复, 疑似存储卡接触不良"),
    ("2026-04-02", "S7-1200", "以太网通信中断, E410", "更换交换机网口, 通信恢复"),
    ("2026-04-18", "FX3U", "上电后 PROG-E 灯闪烁, 报 6801", "程序中存在未闭合的 MC/MCR 指令对, 修正后恢复"),
    ("2026-05-20", "G120变频器", "F0001 过流", "电机轴承卡死导致, 更换轴承"),
    ("2026-05-28", "G120变频器", "F0070 通信超时", "通信电缆屏蔽层未接地导致干扰, 重新接地后恢复"),
    ("2026-06-07", "发那科 LR Mate", "SRVO-062 电池报警", "更换 4 节 D 型电池并执行 Mastering, 位置恢复正常"),
    ("2026-06-15", "S7-1200", "扫描周期超时 E101", "客户程序中 FOR 循环上限被写成变量且溢出, 加上限保护"),
]


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.FAULT_DB)
    conn.executescript("""
        DROP TABLE IF EXISTS fault_codes;
        DROP TABLE IF EXISTS repair_history;
        CREATE TABLE fault_codes (code TEXT PRIMARY KEY, device TEXT, meaning TEXT, action TEXT);
        CREATE TABLE repair_history (date TEXT, device TEXT, fault TEXT, solution TEXT);
    """)
    conn.executemany("INSERT INTO fault_codes VALUES (?,?,?,?)", FAULT_CODES)
    conn.executemany("INSERT INTO repair_history VALUES (?,?,?,?)", REPAIR_HISTORY)
    conn.commit()
    conn.close()
    print(f"[done] {config.FAULT_DB} — {len(FAULT_CODES)} 故障码, {len(REPAIR_HISTORY)} 条维修记录")


if __name__ == "__main__":
    main()
