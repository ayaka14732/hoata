import * as SelectPrimitive from '@radix-ui/react-select'
import { Check, ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  label: string
  value: string
  options: { value: string; label: string }[]
  onChange: (value: string) => void
  dir: 'ltr' | 'rtl'
  placeholder?: string
  className?: string
  testId?: string
  disabled?: boolean
}

export function Select({ label, value, options, onChange, dir, placeholder, className = '', testId, disabled }: Props) {
  return <SelectPrimitive.Root value={value} onValueChange={onChange} dir={dir} disabled={disabled}>
    <SelectPrimitive.Trigger className={`select-trigger ${className}`} aria-label={label} data-testid={testId}>
      <span className="select-value"><SelectPrimitive.Value placeholder={placeholder} /></span>
      <SelectPrimitive.Icon asChild><ChevronDown size={12} /></SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
    <SelectPrimitive.Portal>
      <SelectPrimitive.Content className="select-menu" position="popper" sideOffset={5} align="start" collisionPadding={8}>
        <SelectPrimitive.ScrollUpButton className="select-scroll"><ChevronUp size={13} /></SelectPrimitive.ScrollUpButton>
        <SelectPrimitive.Viewport className="select-viewport">
          {options.map(option => <SelectPrimitive.Item className="select-item" value={option.value} key={option.value} textValue={option.label} title={option.label}>
            <SelectPrimitive.ItemText><bdi className="select-label">{option.label}</bdi></SelectPrimitive.ItemText>
            <SelectPrimitive.ItemIndicator className="select-indicator"><Check size={13} /></SelectPrimitive.ItemIndicator>
          </SelectPrimitive.Item>)}
        </SelectPrimitive.Viewport>
        <SelectPrimitive.ScrollDownButton className="select-scroll"><ChevronDown size={13} /></SelectPrimitive.ScrollDownButton>
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  </SelectPrimitive.Root>
}
