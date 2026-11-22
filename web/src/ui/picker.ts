/**
 * The system file picker. One hidden `<input type="file">` lives in App.tsx and hands itself over with
 * `attachPicker`; anything in the app can then open the picker with `pickFiles`.
 */

let input: HTMLInputElement | null = null;

export function attachPicker(el: HTMLInputElement | null): void {
  input = el;
}

export function pickFiles(): void {
  input?.click();
}
