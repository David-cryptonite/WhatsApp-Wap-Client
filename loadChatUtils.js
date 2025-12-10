// =================== CHAT HISTORY UTILITIES (ESM) ===================
// Module-level dependencies - will be set by serverWml.js
let logger = console; // Default to console
let sock = null;
let chatStore = null;
let messageStore = null;
let connectionState = 'disconnected';
let formatJid = (jid) => jid;
let delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
let saveMessageToDB = () => {};
let performInitialSync = async () => {};

// Flag to show warning only once
let chatHistoryWarningShown = false;

// Initialize dependencies from main server
function initializeDependencies(deps) {
  logger = deps.logger || console;
  sock = deps.sock;
  chatStore = deps.chatStore;
  messageStore = deps.messageStore;
  connectionState = deps.connectionState;
  formatJid = deps.formatJid;
  delay = deps.delay;
  saveMessageToDB = deps.saveMessageToDB;
  performInitialSync = deps.performInitialSync;
}

// Getter for dynamic sock access
function getSock() {
  return sock;
}

function getConnectionState() {
  return connectionState;
}

async function loadChatHistory(jid, batchSize = 99999999999999) {
  const currentSock = getSock();
  if (!currentSock) {
    logger.warn('Cannot load chat history: socket not available');
    return [];
  }

  try {
    const formattedJid = formatJid(jid);

    // FIRST: Try to load from local chatStore (messages already synced by Baileys)
    if (chatStore && chatStore.has(formattedJid)) {
      const chatMessages = chatStore.get(formattedJid) || [];
      if (chatMessages.length > 0) {
        logger.info(`Loaded ${chatMessages.length} messages for ${jid} from local store`);
        return chatMessages.slice(0, batchSize);
      }
    }

    // SECOND: Try to fetch from WhatsApp if methods are available
    if (!currentSock.fetchMessagesFromWA && !currentSock.loadMessages) {
      // Show warning only once to avoid log spam
      if (!chatHistoryWarningShown) {
        logger.info('Chat history will load automatically from WhatsApp sync (messaging-history.set event)');
        chatHistoryWarningShown = true;
      }
      return [];
    }

    // THIRD: If Baileys has fetch methods, try to use them
    logger.info(`Fetching chat history from WhatsApp for ${jid} (max ${batchSize} messages)`);

    let cursor = undefined;
    let keepFetching = true;
    let allMessages = [];
    let fetchCount = 0;
    const maxFetches = Math.ceil(batchSize / 9999); // Optimized for 4GB RAM

    while (keepFetching && fetchCount < maxFetches && allMessages.length < batchSize) {
      let batch;

      // Fetch in larger chunks for 4GB Raspberry Pi 4
      const chunkSize = Math.min(9999, batchSize - allMessages.length);

      // Try the available method
      if (currentSock.loadMessages) {
        batch = await currentSock.loadMessages(formattedJid, chunkSize, cursor);
      } else if (currentSock.fetchMessagesFromWA) {
        batch = await currentSock.fetchMessagesFromWA(formattedJid, chunkSize, cursor);
      }

      if (!batch || batch.length === 0) break;

      for (const msg of batch) {
        if (!msg.key?.id) continue;
        if (allMessages.length >= batchSize) break; // Stop if limit reached
        saveMessageToDB(msg, formattedJid);
        allMessages.push(msg);
      }

      cursor = batch[0]?.key;
      fetchCount++;
     // await delay(300); // Faster for 4GB system
    }

    logger.info(`Finished loading history for ${jid}, total: ${allMessages.length} messages`);
    return allMessages;
  } catch (err) {
    logger.error(`Failed to load chat history for ${jid}:`, err.message);
    return [];
  }
}




// =================== FUNZIONI DI SUPPORTO ===================

async function loadAllChatsHistory(maxChatsToLoad = 50, messagesPerChat = 100) {
  const currentSock = getSock();
  const currentState = getConnectionState();

  if (!currentSock || currentState !== 'open') {
    logger.warn('Cannot load all chats history: not connected');
    return false;
  }

  try {
    if (!chatStore || !chatStore.keys) {
      logger.warn('chatStore not available');
      return false;
    }

    const chatIds = Array.from(chatStore.keys()).slice(0, maxChatsToLoad);
    logger.info(`Chat history already available for ${chatIds.length} chats from local store`);

    let totalMessages = 0;
    for (const chatId of chatIds) {
      const chatMessages = chatStore.get(chatId) || [];
      totalMessages += chatMessages.length;
    }

    logger.info(`Total messages in local store: ${totalMessages} across ${chatIds.length} chats`);
    return true;

  } catch (error) {
    logger.error('Bulk chat history load failed:', error.message);
    return false;
  }
}

async function loadRecentMessages(jid, hours = 24) {
  const cutoffTime = Math.floor((Date.now() - (hours * 60 * 60 * 1000)) / 1000);

  try {
    const formattedJid = formatJid(jid);

    // Load from local chatStore (already synced by Baileys)
    if (chatStore && chatStore.has(formattedJid)) {
      const allMessages = chatStore.get(formattedJid) || [];

      // Filter by time if needed
      const recentMessages = allMessages.filter(msg => {
        const msgTime = Number(msg.messageTimestamp) || 0;
        return msgTime >= cutoffTime;
      });

      logger.info(`Found ${recentMessages.length} recent messages for ${jid} (last ${hours} hours)`);
      return recentMessages;
    }

    logger.info(`No messages found for ${jid} in local store`);
    return [];

  } catch (error) {
    logger.error(`Failed to load recent messages for ${jid}:`, error.message);
    return [];
  }
}

async function preloadImportantChats() {
  const currentSock = getSock();
  if (!currentSock) return;

  try {
    if (!chatStore || !chatStore.entries) {
      logger.warn('chatStore not available for preloading');
      return;
    }

    // Identifica chat importanti (con molti messaggi o attività recente)
    const importantChats = Array.from(chatStore.entries())
      .map(([jid, messages]) => ({
        jid,
        messageCount: messages.length,
        lastActivity: messages.length > 0 ?
          Math.max(...messages.map(m => Number(m.messageTimestamp))) : 0
      }))
      .sort((a, b) => b.lastActivity - a.lastActivity)
      .slice(0, 20); // Top 20 most active chats

    let totalMessages = 0;
    for (const chat of importantChats) {
      totalMessages += chat.messageCount;
    }

    logger.info(`Important chats already loaded: ${importantChats.length} chats with ${totalMessages} messages`);

  } catch (error) {
    logger.error('Important chats preload failed:', error.message);
  }
}

// Integrazione nel sistema esistente
async function enhancedInitialSync() {
  try {
    logger.info('Starting enhanced sync...');

    // Prima esegui la sync base
    if (performInitialSync) {
      await performInitialSync();
    }

    // Chat histories are automatically loaded by Baileys via messaging-history.set event
    if (chatStore && chatStore.size > 0) {
      const totalChats = chatStore.size;
      let totalMessages = 0;

      for (const [jid, messages] of chatStore.entries()) {
        totalMessages += messages.length;
      }

      logger.info(`Chat history loaded: ${totalChats} chats with ${totalMessages} messages (from Baileys sync)`);

      // Report on important chats
      await preloadImportantChats();

      logger.info('Enhanced sync completed successfully');
    } else {
      logger.info('No chat history available yet - waiting for Baileys messaging-history.set event');
    }

  } catch (error) {
    logger.error('Enhanced sync failed:', error.message);
  }
}

// Export functions (ESM)
export {
  initializeDependencies,
  getSock,
  getConnectionState,
  loadChatHistory,
  loadAllChatsHistory,
  loadRecentMessages,
  preloadImportantChats,
  enhancedInitialSync
};
