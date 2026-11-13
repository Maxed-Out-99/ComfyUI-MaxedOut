// Run with node scripts/check_set_get.mjs. No server or personal files needed.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../web/nodes/mxd_set_get.js', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '');
const run = context => vm.runInContext(`(async () => {${source}\n})()`, context);

function environment(extensions = [], existing = {}) {
  const registered = [];
  class Node {
    constructor() { this.properties = {}; this.inputs = []; this.outputs = []; this.widgets = []; }
    addInput(name, type) { this.inputs.push({ name, type, link: null }); }
    addOutput(name, type) { this.outputs.push({ name, type, links: [] }); }
    addWidget(type, name, value, callback, options) {
      const widget = { type, name, value, callback, options };
      this.widgets.push(widget); return widget;
    }
    serialize() { return { type: this.type, widgets_values: this.widgets.map(w => w.value) }; }
  }
  const LiteGraph = {
    LGraphNode: Node, registered_node_types: { ...existing },
    registerNodeType(key, cls) {
      assert.ok(!this.registered_node_types[key], `Duplicate ${key}`);
      cls.prototype.type = key; this.registered_node_types[key] = cls;
    }
  };
  const context = vm.createContext({
    LiteGraph, window: {}, console, setTimeout, clearTimeout,
    api: { getExtensions: async () => extensions },
    app: {
      registerExtension: ext => registered.push(ext),
      canvas: { setDirty() {} },
      ui: { settings: { getSettingValue() { return false; } } }
    }
  });
  return { context, registered, LiteGraph };
}

// Detection is based on the served file list, even before KJ runs its code.
for (const url of [
  '/extensions/comfyui-kjnodes/js/setgetnodes.js',
  '/extensions/renamed%20KJ%20pack/js/setgetnodes.js?version=2'
]) {
  const env = environment([url]);
  await run(env.context);
  assert.equal(env.registered.length, 0);
  assert.equal(env.context.window.kjNodes, undefined);
}
const existing = environment([], { SetNode: class {}, GetNode: class {} });
await run(existing.context);
assert.equal(existing.registered.length, 0);
const failed = environment();
failed.context.api.getExtensions = async () => { throw new Error('simulated offline'); };
failed.context.console = { warn() {} };
await run(failed.context);
assert.equal(failed.registered.length, 0);

// Two MXD installations racing must install only one provider.
const env = environment(['/extensions/ComfyUI-MaxedOut/nodes/mxd_set_get.js']);
await Promise.all([run(env.context), run(env.context)]);
assert.equal(new Set(env.registered.map(e => e.name)).size, env.registered.length);
assert.equal(env.registered.filter(e => e.name === 'KJNodes.SetNode').length, 1);
for (const ext of env.registered) ext.registerCustomNodes?.();
assert.deepEqual(Object.keys(env.LiteGraph.registered_node_types).sort(), ['GetNode', 'SetNode']);
const { SetNode, GetNode } = env.LiteGraph.registered_node_types;
const setter = new SetNode();
const getter = new GetNode();
setter.widgets[0].value = getter.widgets[0].value = 'VAE';
setter.inputs[0] = { name: 'VAE', type: 'VAE', link: 7 };
const upstream = { id: 1, type: 'VAELoader' };
const link = { id: 7, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0, type: 'VAE' };
const graph = { _nodes: [upstream, setter, getter], _links: { 7: link }, getNodeById: id => id === 1 ? upstream : undefined };
setter.graph = getter.graph = graph;
getter.onRename();
assert.equal(getter.findSetter(graph), setter);
assert.equal(getter.outputs[0].type, 'VAE');
assert.equal(getter.getInputLink(0), link);
assert.equal(getter.serialize().widgets_values[0], 'VAE');
// Renaming a setter must update the paired getter's saved name.
setter.properties.previousName = 'VAE';
setter.widgets[0].value = 'Shared VAE';
setter.update();
assert.equal(getter.widgets[0].value, 'Shared VAE');
assert.equal(getter.getInputLink(0), link);
console.log('OK: Set/Get provider coexistence, singleton, routing, serialization and rename checks');
