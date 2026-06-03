"""类型强制转换 mixin 模块。

提供一组用于 LLM 返回值验证和类型转换的工具方法。
由于大语言模型的返回值类型不确定，需要对整数、浮点数、布尔值、
字符串列表等进行严格的类型检查和转换，确保下游数据一致性。
"""

from __future__ import annotations

from typing import Any


class CoercionMixin:
    """类型强制转换混入类，为 LLM 结果验证提供共享的类型转换辅助方法。

    该类设计为 mixin 模式，由需要解析 LLM 返回值的客户端类继承使用。
    所有转换方法在类型不匹配时会抛出 ValueError，错误信息中包含
    字段名和实际返回值，便于调试。
    """

    def _coerce_int(self, value: Any, field_name: str) -> int:
        """将任意值强制转换为整数类型。

        布尔值（True/False）被视为无效输入并拒绝转换，
        因为在 Python 中 bool 是 int 的子类（True==1, False==0），
        直接转换会导致语义混淆。

        Args:
            value: 待转换的值，来自 LLM 返回的 JSON
            field_name: 字段名称，用于生成错误提示信息

        Returns:
            转换后的整数值

        Raises:
            ValueError: 当值为 bool 类型或无法转换为 int 时
        """
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}") from exc

    def _coerce_float(self, value: Any, field_name: str) -> float:
        """将任意值强制转换为浮点数类型。

        同样拒绝布尔值输入，原因同 _coerce_int。

        Args:
            value: 待转换的值，来自 LLM 返回的 JSON
            field_name: 字段名称，用于生成错误提示信息

        Returns:
            转换后的浮点数值

        Raises:
            ValueError: 当值为 bool 类型或无法转换为 float 时
        """
        if isinstance(value, bool):
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} from LLM: {value!r}") from exc

    def _coerce_bool(self, value: Any, field_name: str) -> bool:
        """验证并返回布尔值。

        仅接受真正的 bool 类型输入，不支持从字符串或数字隐式转换，
        以避免 LLM 返回模糊值时产生误判。

        Args:
            value: 待验证的值，来自 LLM 返回的 JSON
            field_name: 字段名称，用于生成错误提示信息

        Returns:
            原样返回的布尔值

        Raises:
            ValueError: 当值不是 bool 类型时
        """
        if isinstance(value, bool):
            return value
        raise ValueError(f"Invalid {field_name} from LLM: {value!r}")

    def _coerce_str_list(self, value: Any, field_name: str) -> list[str]:
        """将值转换为去空格后的字符串列表。

        遍历列表中的每个元素，转换为字符串并去除首尾空白，
        同时过滤掉转换后为空字符串的元素。

        Args:
            value: 待转换的值，必须为 list 类型
            field_name: 字段名称，用于生成错误提示信息

        Returns:
            去除空元素和多余空白后的字符串列表

        Raises:
            ValueError: 当值不是 list 类型时
        """
        if not isinstance(value, list):
            raise ValueError(f"Invalid {field_name} from LLM: expected list")
        return [str(item).strip() for item in value if str(item).strip()]

    def _clamp01(self, value: float) -> float:
        """将浮点数限制在 [0, 1] 范围内，保留 4 位小数。

        用于置信度等概率值的边界裁剪，防止 LLM 返回超出范围的值。

        Args:
            value: 待限制的浮点数。

        Returns:
            限制后的浮点数。
        """
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return round(value, 4)
