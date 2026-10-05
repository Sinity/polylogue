// @vitest-environment node
import { readFileSync } from "node:fs";
import { Script, createContext } from "node:vm";
import assert from "node:assert/strict";
import { it } from "vitest";
import { restoreReceiverConfiguration } from "../scripts/live_provider_proof.mjs";
const source=readFileSync(new URL('../src/background/runtime.js',import.meta.url),'utf8');
const definitions=source.slice(source.indexOf('async function receiverSettings()'),source.indexOf('function hostnameForUrl('));
const health=source.slice(source.indexOf('async function pairWithCode('),source.indexOf('async function appendCaptureLog('));
const handler=source.slice(source.indexOf('    if (message.type === "polylogue.configureReceiver")'),source.indexOf('    if (message.type === "polylogue.pairWithCode")'));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
function fixture({blockPairing=false}={}){
 const storage={receiverBaseUrl:'http://127.0.0.1:41234',receiverAuthToken:'neutral-token'};
 const probe=deferred(),entered=deferred(),write=deferred(),written=deferred();const writes=[];
 const context=createContext({Date,URL,RECEIVER_PAIRING_KEY:'polylogueReceiverPairing',RECEIVER_API_SCHEMA:'polylogue-browser-capture/v1',DEFAULT_RECEIVER:'http://127.0.0.1:8765',NATIVE_BOOTSTRAP_HOST:'neutral-host',trustedReceiverHealthCache:null,runtimeChrome:{permissions:{contains:async()=>true},storage:{local:{async get(defaults){const keys=Array.isArray(defaults)?defaults:Object.keys(defaults);return Object.fromEntries(keys.filter(k=>Object.hasOwn(storage,k)||!Array.isArray(defaults)).map(k=>[k,Object.hasOwn(storage,k)?storage[k]:defaults[k]]));},async set(values){if(blockPairing&&values.polylogueReceiverPairing){written.resolve();await write.promise;}Object.assign(storage,values);writes.push(Object.keys(values));},async remove(keys){for(const key of Array.isArray(keys)?keys:[keys])delete storage[key];}}}},probeReceiverStatus:async()=>{entered.resolve();return probe.promise;}});
 new Script('let storageMutationQueue = Promise.resolve();'+source.slice(source.indexOf('function serializeStorageMutation('),source.indexOf('function replaceLegacyAcceptedMessageIdentities('))+definitions+health+'\nglobalThis.scope=receiverHealthScope;globalThis.health=checkReceiverHealth;globalThis.pair=pairWithCode;globalThis.cache=()=>trustedReceiverHealthCache;globalThis.restore=restoreReceiverSettings;globalThis.configure=saveReceiverSettings;globalThis.reset=clearReceiverPairing;globalThis.dispatch=async function(message,sendResponse){'+handler+'};').runInContext(context);
 return {context,storage,probe,entered,write,written,writes};
}

it("serializes receiver restoration against delayed status, code and cache publication", async () => {
const owned={baseUrl:'http://127.0.0.1:41234',token:'neutral-token',receiverId:null,revision:0};
const response={body:{ok:true,receiver_id:'neutral-receiver',api_schema:'polylogue-browser-capture/v1'},response:{ok:true,status:200}};

// Original old health response completes after runtime-owned prior-absent restore.
{
 const f=fixture();const pending=f.context.health({allowCanonicalRecovery:false,allowCredentialRefresh:false});await f.entered.promise;
 await f.context.restore({},owned);f.probe.resolve(response);const result=await pending;
 assert.equal(result.detail,'receiver_configuration_changed');assert.deepEqual(Object.keys(f.storage),[]);
}
// Original pairing write is physically suspended: restoration queues behind it.
{
 const f=fixture({blockPairing:true});const pending=f.context.health({allowCanonicalRecovery:false,allowCredentialRefresh:false});await f.entered.promise;f.probe.resolve(response);await f.written.promise;
 let settled=false;const restoring=f.context.restore({},owned).then(()=>{settled=true;});await Promise.resolve();await Promise.resolve();assert.equal(settled,false);assert.equal(f.storage.receiverAuthToken,'neutral-token');
 f.write.resolve();await Promise.all([pending,restoring]);assert.deepEqual(Object.keys(f.storage),[]);
}
// Same-values configuration and ordinary reset invalidate already-started probes.
for(const mutation of ['configure','reset']){
 const f=fixture();const pending=f.context.health({allowCanonicalRecovery:false,allowCredentialRefresh:false});await f.entered.promise;
 if(mutation==='configure')await f.context.configure(owned.baseUrl,owned.token);else await f.context.reset();
 f.probe.resolve(response);assert.equal((await pending).detail,'receiver_configuration_changed');assert.notEqual(f.storage.polylogueReceiverPairing?.state,'online');
}
// Foreign configuration is preserved; restore refuses rather than overwrites.
{
 const f=fixture();f.storage.receiverAuthToken='neutral-other-owner';const before=JSON.stringify(f.storage);await assert.rejects(f.context.restore({},owned),/proof_receiver_configuration_changed/);assert.equal(JSON.stringify(f.storage),before);
}
// Exact original proof CDP expression reaches actual configure handler, not storage.
{
 const f=fixture();let pause=false;const client={async call(method,params){assert.equal(method,'Runtime.evaluate');let response;const chrome={runtime:{async sendMessage(message){if(message.type==='polylogue.ambient.configure'){pause=true;return {ok:true};}await f.context.dispatch(message,value=>{response=value;});return response;}}};const value=await new Script(params.expression).runInContext(createContext({chrome}));return {result:{value}};}};
 await restoreReceiverConfiguration(client,{},owned);assert.equal(pause,true);assert.deepEqual(Object.keys(f.storage),[]);
}
// Ordinary 401 credential refresh remains valid under the newly written scope.
{
 const f=fixture();let probes=0;
 f.context.runtimeChrome.runtime={id:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',sendNativeMessage:async()=>({ok:true,auth_token:'neutral-refreshed',receiver_id:'neutral-receiver',api_schema:'polylogue-browser-capture/v1'})};
 f.context.probeReceiverStatus=async()=>++probes===1?{body:{error:'unauthorized'},response:{status:401}}:response;
 const result=await f.context.health({allowCanonicalRecovery:false});assert.equal(result.status,'ok');assert.equal(probes,2);assert.equal(f.storage.receiverAuthToken,'neutral-refreshed');
}
// A delayed native credential result cannot undo owned configuration removal.
{
 const f=fixture();f.storage.receiverAuthToken='';const native=deferred(),start=deferred();
 f.context.runtimeChrome.runtime={id:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',sendNativeMessage:async()=>{start.resolve();return native.promise;}};
 const pending=f.context.health();await start.promise;await f.context.restore({}, {...owned,token:''});
 native.resolve({ok:true,auth_token:'neutral-late',receiver_id:'neutral-receiver',api_schema:'polylogue-browser-capture/v1'});
 assert.equal((await pending).detail,'receiver_configuration_changed');assert.deepEqual(Object.keys(f.storage),[]);
}
// Ordinary canonical endpoint recovery keeps its actual receiver identity law.
{
 const f=fixture();f.storage.polylogueReceiverPairing={receiver_id:'neutral-receiver',api_schema:'polylogue-browser-capture/v1',endpoint:'http://127.0.0.1:41234',dev_override:false};
 f.context.probeReceiverStatus=async endpoint=>{if(endpoint===owned.baseUrl)throw Error('neutral-offline');return response;};
 assert.equal((await f.context.health({allowCredentialRefresh:false})).status,'recovered');assert.equal(f.storage.receiverBaseUrl,'http://127.0.0.1:8765');assert.equal(f.storage.polylogueReceiverPairing.state,'online');
}
// Restoration retains the optional-origin enforcement and typed refusal.
{
 const f=fixture();f.context.runtimeChrome.permissions.contains=async()=>false;const before=JSON.stringify(f.storage);
 await assert.rejects(f.context.restore({receiverBaseUrl:'http://127.0.0.1:41235'},owned),/receiver_origin_not_permitted/);assert.equal(JSON.stringify(f.storage),before);
 await assert.rejects(f.context.restore({receiverAuthToken:[]},owned),/proof_receiver_configuration_changed/);assert.equal(JSON.stringify(f.storage),before);
}
// Malformed explicit restore never becomes an ordinary default configuration.
{
 const f=fixture();const before=JSON.stringify(f.storage);
 for (const restore of [null, false, [], "neutral-private", {}]) {
   let result;await f.context.dispatch({type:'polylogue.configureReceiver',restore},value=>{result=value;});
   assert.equal(result.ok,false);assert.equal(result.error,'proof_receiver_configuration_changed');assert.equal(JSON.stringify(f.storage),before);
 }

}
// Delayed code exchange is bound to the original captured configuration.
for (const mutation of ['restore','configure','reset']) {
 const f=fixture();const redeem=deferred(),started=deferred();
 f.context.redeemPairingCode=async()=>{started.resolve();return redeem.promise;};
 const pending=f.context.pair('neutral-code');await started.promise;
 if(mutation==='restore')await f.context.restore({},owned);else if(mutation==='configure')await f.context.configure(owned.baseUrl,owned.token);else await f.context.reset();
 const before=JSON.stringify(f.storage);redeem.resolve({ok:true,authToken:'neutral-redeemed',receiverId:'neutral-receiver'});
 assert.equal((await pending).error,'receiver_configuration_changed');assert.equal(JSON.stringify(f.storage),before);
}
// Ordinary successful code exchange still confirms the canonical health.
{
 const f=fixture();f.context.redeemPairingCode=async()=>({ok:true,authToken:'neutral-redeemed',receiverId:'neutral-receiver'});f.context.probeReceiverStatus=async()=>response;
 const result=await f.context.pair('neutral-code');assert.equal(result.ok,true);assert.equal(result.health.status,'ok');assert.equal(f.storage.receiverAuthToken,'neutral-redeemed');assert.equal(f.storage.polylogueReceiverPairing.receiver_id,'neutral-receiver');
}
// A configuration waiting for its permission check must precede cache publication.
{
 const f=fixture();const permission=deferred(),configStarted=deferred();let changing;
 f.context.runtimeChrome.permissions.contains=async()=>{configStarted.resolve();return permission.promise;};
 const set=f.context.runtimeChrome.storage.local.set;
 f.context.runtimeChrome.storage.local.set=async values=>{await set(values);if(values.polylogueReceiverPairing?.state==='online')changing=f.context.configure('http://127.0.0.1:41235','neutral-next-token');};
 const pending=f.context.health({allowCanonicalRecovery:false,allowCredentialRefresh:false});await f.entered.promise;f.probe.resolve(response);await configStarted.promise;
 assert.equal(f.context.cache(),null);permission.resolve(true);await changing;
 assert.equal((await pending).detail,'receiver_configuration_changed');assert.equal(f.context.cache(),null);assert.equal(f.storage.receiverAuthToken,'neutral-next-token');
}

// Bootstrap confirmation retains its original scope across recursive health entry.
{
 const f=fixture();f.storage.receiverAuthToken='';let probes=0;
 f.context.probeReceiverStatus=async()=>{probes++;return response;};
 f.context.bootstrapReceiverCredential=async()=>{
   const configured=await f.context.configure(owned.baseUrl,'neutral-confirmed',await f.context.scope());
   await f.context.restore({}, {...owned,token:'neutral-confirmed',revision:configured.revision});
   return {ok:true,scope:configured};
 };
 assert.equal((await f.context.health()).detail,'receiver_configuration_changed');
 assert.equal(probes,0);assert.deepEqual(Object.keys(f.storage),[]);
}

// Exact values do not confer ownership after an explicit reset/configure.
for(const mutation of ['configure','reset']) {
 const f=fixture();const previous={receiverBaseUrl:'http://127.0.0.1:8765',receiverAuthToken:'neutral-original',polylogueReceiverPairing:{receiver_id:'neutral-original'}};
 const admitted=await f.context.configure(owned.baseUrl,owned.token);
 const admittedOwned={...owned,revision:admitted.configurationRevision};
 if(mutation==='configure') await f.context.configure(owned.baseUrl,owned.token);else await f.context.reset();
 const before=JSON.stringify(f.storage);
 await assert.rejects(f.context.restore(previous,admittedOwned),/proof_receiver_configuration_changed/);
 assert.equal(JSON.stringify(f.storage),before);
}

});

it("retains admitted configure/reset scope through proof handshake and cleanup", async () => {
 const { receiverConfigurationOwner } = await import('./infra/receiver_configuration.js');
 const { proofReceiverCustody, configureProofReceiver, cleanupProofReceiver } = await import('../scripts/live_provider_proof.mjs');
 for(const schedule of ['success','unreachable','reset','configure','suspended']) {
  const previous={receiverBaseUrl:'http://127.0.0.1:8765',receiverAuthToken:'neutral-original',polylogueReceiverPairing:{receiver_id:'neutral-original'}};
  const values=structuredClone(previous), pending=deferred(), entered=deferred();
  const chrome={permissions:{contains:async()=>true},storage:{local:{
   async get(defaults){const keys=Array.isArray(defaults)?defaults:Object.keys(defaults);return Object.fromEntries(keys.filter(k=>Object.hasOwn(values,k)||!Array.isArray(defaults)).map(k=>[k,Object.hasOwn(values,k)?values[k]:defaults[k]]));},
   async set(rows){Object.assign(values,rows);},async remove(keys){for(const key of Array.isArray(keys)?keys:[keys])delete values[key];}
  }}};
  const originalOwner=receiverConfigurationOwner(chrome,async()=>{
   entered.resolve();if(schedule==='unreachable')throw Error('neutral-offline');
   if(schedule==='suspended')await pending.promise;
   return {response:{ok:true,status:200},body:{ok:true,receiver_id:'neutral-proof',api_schema:'polylogue-browser-capture/v1'}};
  });
  chrome.runtime={async sendMessage(message){
   if(message.type==='polylogue.ambient.configure')return {ok:true};
   if(message.type==='polylogue.receiverPairing.reset'){
    if(schedule==='reset')await originalOwner.reset();
    if(schedule==='configure')await originalOwner.send({type:'polylogue.configureReceiver',receiverBaseUrl:'http://127.0.0.1:41234',receiverAuthToken:'neutral-proof-token'});
   }
   return originalOwner.send(message);
  }};
  const client={async call(_method,params){return {result:{value:await new Script(params.expression).runInContext(createContext({chrome}))}};}};
  const owner=proofReceiverCustody(client,previous,{baseUrl:'http://127.0.0.1:41234',token:'neutral-proof-token',receiverId:null},'http://127.0.0.1:41234/*');
  const configuring=configureProofReceiver(owner);let primary;
  if(schedule==='suspended'){
   await entered.promise;let settled=false;const cleaning=cleanupProofReceiver(owner).then(()=>{settled=true;});
   await Promise.resolve();await Promise.resolve();assert.equal(settled,false);assert.equal(values.receiverAuthToken,'neutral-proof-token');
   pending.resolve();await configuring;await cleaning;
  } else {
   try{await configuring;}catch(error){primary=error;}
   if(['reset','configure'].includes(schedule)){
    assert(primary);const before=JSON.stringify(values);await assert.rejects(cleanupProofReceiver(owner),/proof_receiver_cleanup_failed/);assert.equal(JSON.stringify(values),before);if(schedule==='reset')assert.notEqual(values.polylogueReceiverPairing?.receiver_id,'neutral-original');continue;
   }
   if(schedule==='unreachable')assert.equal(primary?.message,'proof_receiver_handshake_failed');else assert.equal(primary,undefined);
   assert.equal(owner.owned.revision,2);
   if(schedule==='unreachable')await assert.rejects(cleanupProofReceiver(owner),/proof_receiver_cleanup_failed/);else await cleanupProofReceiver(owner);
  }
  assert.deepEqual(JSON.parse(JSON.stringify(values)),previous);
  assert.equal(owner.cleanup.receiver,'settled');
 }
});
