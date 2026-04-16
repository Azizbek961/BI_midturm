from django import template


register = template.Library()


@register.filter
def get_item(mapping, key):
    if isinstance(mapping, dict):
        return mapping.get(key, '')
    return ''


@register.filter
def format_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value

    if number.is_integer():
        return f'{int(number):,}'
    return f'{number:,.2f}'
