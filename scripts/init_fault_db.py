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
    ("E203", "S7-1200", "CPU 内部错误: 程序块校验失败", "1.重新下载程序 2.检查存储卡 3.仍报错则恢复出厂设置"),
    ("F0001", "西门子变频器 G120", "过流故障", "1.检查电机电缆短路 2.检查加速时间是否过短 3.检查负载是否卡死"),
    ("F0002", "西门子变频器 G120", "过压故障", "1.检查电源电压 2.延长减速时间 3.加装制动电阻"),
    ("E-STOP", "发那科机器人 LR Mate", "急停回路触发", "1.检查急停按钮是否复位 2.检查安全门信号 3.检查外部急停接线"),
    ("SRVO-023", "发那科机器人", "伺服过热", "1.降低负载/速度 2.检查散热风扇 3.检查环境温度是否超40°C"),
]

REPAIR_HISTORY = [
    ("2026-03-12", "S7-1200", "ERROR灯常亮, E203", "重新下载程序后恢复, 疑似存储卡接触不良"),
    ("2026-04-02", "S7-1200", "以太网通信中断", "更换交换机网口, 通信恢复"),
    ("2026-05-20", "G120变频器", "F0001 过流", "电机轴承卡死导致, 更换轴承"),
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
