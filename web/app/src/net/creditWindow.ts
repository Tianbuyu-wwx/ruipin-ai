/**
 * 信用窗口（方案 §2.4 背压）—— 对应 Python `protocol.CreditWindow`。
 *
 * 限制客户端"在途未消费"的媒体帧数：服务端每消费 N 帧回一个 `media.ack`，
 * 客户端窗口未满才继续发，防弱网堆积。窗口满时 `tryAcquire()` 返回 false，
 * **由调用方决定**丢弃/等待/报错，本类不替业务做兜底决策。
 */

export class CreditWindow {
  private readonly _capacity: number;
  private _used = 0;

  constructor(capacity: number) {
    if (!Number.isInteger(capacity)) {
      throw new Error(`capacity 必须是整数，实际: ${JSON.stringify(capacity)}`);
    }
    if (capacity < 1) {
      throw new Error(`capacity 必须 >= 1，实际: ${capacity}`);
    }
    this._capacity = capacity;
  }

  get capacity(): number {
    return this._capacity;
  }

  /** 在途（已占用未释放）的帧数。 */
  get used(): number {
    return this._used;
  }

  get available(): number {
    return this._capacity - this._used;
  }

  get full(): boolean {
    return this._used >= this._capacity;
  }

  /** 尝试占用 n 个信用。窗口不足时返回 false 且**不占用任何**信用。 */
  tryAcquire(n = 1): boolean {
    assertPositiveInt(n);
    if (this._used + n > this._capacity) return false;
    this._used += n;
    return true;
  }

  /** 释放 n 个信用（服务端已消费）。返回释放后的在途量。 */
  release(n = 1): number {
    assertPositiveInt(n);
    if (n > this._used) {
      throw new Error(`释放量 ${n} 超过在途量 ${this._used}（重复消费，必须显式暴露）`);
    }
    this._used -= n;
    return this._used;
  }

  /** 清空窗口（断线重连用：在途帧已随连接丢失）。返回被释放的帧数。 */
  releaseAll(): number {
    const freed = this._used;
    this._used = 0;
    return freed;
  }
}

function assertPositiveInt(n: number): void {
  if (!Number.isInteger(n) || n < 1) {
    throw new Error(`n 必须是 >= 1 的整数，实际: ${JSON.stringify(n)}`);
  }
}
