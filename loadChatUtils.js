// =================== CHAT HISTORY UTILITIES ===================
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

async function loadChatHistory(jid, batchSize = 50) {
  const currentSock = getSock();
  if (!currentSock) {
    logger.warn('Cannot load chat history: socket not available');
    return [];
  }

  try {
    logger.info(`Loading chat history for ${jid} (max ${batchSize} messages)`);

    const formattedJid = formatJid(jid);

    // Check if the method exists in Baileys
    if (!currentSock.fetchMessagesFromWA && !currentSock.loadMessages) {
      logger.warn('Chat history loading not available in this Baileys version');
      return [];
    }

    let cursor = undefined;
    let keepFetching = true;
    let allMessages = [];
    let fetchCount = 0;
    const maxFetches = Math.ceil(batchSize / 25); // Limit fetches based on batch size

    while (keepFetching && fetchCount < maxFetches && allMessages.length < batchSize) {
      let batch;

      // Fetch in smaller chunks (max 25 at a time for Raspberry Pi)
      const chunkSize = Math.min(25, batchSize - allMessages.length);

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
      await delay(500); // Longer delay to prevent rate limiting
    }

    logger.info(`Finished loading history for ${jid}, total: ${allMessages.length} messages`);
    return allMessages;
  } catch (err) {
    logger.error(`Failed to load chat history for ${jid}:`, err.message);
    return [];
  }
}




// =================== FUNZIONI DI SUPPORTO ===================

async function loadAllChatsHistory(maxChatsToLoad = 20, messagesPerChat = 50) {
  const currentSock = getSock();
  const currentState = getConnectionState();

  if (!currentSock || currentState !== 'open') {
    logger.warn('Cannot load all chats history: not connected');
    return false;
  }

  try {
    logger.info(`Starting bulk chat history load: ${maxChatsToLoad} chats, ${messagesPerChat} messages each`);

    if (!chatStore || !chatStore.keys) {
      logger.warn('chatStore not available');
      return false;
    }

    const chatIds = Array.from(chatStore.keys()).slice(0, maxChatsToLoad);
    let successCount = 0;
    let failCount = 0;

    for (const chatId of chatIds) {
      try {
        const messages = await loadChatHistory(chatId, messagesPerChat);
        if (messages.length > 0) {
          successCount++;
          logger.debug(`Loaded ${messages.length} messages for ${chatId}`);
        }

        // Longer delay to prevent rate limiting and reduce Raspberry Pi load
        await delay(2000);

      } catch (chatError) {
        failCount++;
        logger.error(`Failed to load history for ${chatId}:`, chatError.message);
      }
    }

    logger.info(`Bulk history load complete: ${successCount} success, ${failCount} failed`);
    return successCount > 0;

  } catch (error) {
    logger.error('Bulk chat history load failed:', error.message);
    return false;
  }
}

async function loadRecentMessages(jid, hours = 24) {
  const cutoffTime = Math.floor((Date.now() - (hours * 60 * 60 * 1000)) / 1000);

  try {
    // Load only 50 recent messages (Raspberry Pi friendly)
    const messages = await loadChatHistory(jid, 50);

    logger.info(`Found ${messages.length} messages for ${jid}`);
    return messages;

  } catch (error) {
    logger.error(`Failed to load recent messages for ${jid}:`, error.message);
    return [];
  }
}

async function preloadImportantChats() {
  const currentSock = getSock();
  if (!currentSock) return;

  try {
    logger.info('Preloading important chats...');

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
      .slice(0, 5); // Top 5 chat più attive (reduced for Raspberry Pi)

    for (const chat of importantChats) {
      try {
        // Load only 30 messages per important chat (Raspberry Pi friendly)
        await loadChatHistory(chat.jid, 30);
        await delay(3000); // Longer delay to reduce Raspberry Pi load
      } catch (error) {
        logger.warn(`Failed to preload important chat ${chat.jid}:`, error.message);
      }
    }

    logger.info(`Preloaded ${importantChats.length} important chats`);

  } catch (error) {
    logger.error('Important chats preload failed:', error.message);
  }
}

// Integrazione nel sistema esistente
async function enhancedInitialSync() {
  try {
    logger.info('Starting enhanced sync with message history loading (Raspberry Pi optimized)...');

    // Prima esegui la sync base
    if (performInitialSync) {
      await performInitialSync();
    }

    // Poi carica la cronologia dei messaggi con limiti per Raspberry Pi
    if (chatStore && chatStore.size > 0) {
      logger.info('Loading chat histories (limited for Raspberry Pi)...');

      // Carica cronologia solo per 10 chat con 30 messaggi ciascuna
      // Questo previene l'esaurimento della memoria su Raspberry Pi
      await loadAllChatsHistory(10, 30);

      // Precarica solo le 5 chat più importanti
      await preloadImportantChats();

      logger.info('Enhanced sync completed successfully (Raspberry Pi mode)');
    }

  } catch (error) {
    logger.error('Enhanced sync failed:', error.message);
  }
}

// Esporta le funzioni
module.exports = {
  initializeDependencies,
  getSock,
  getConnectionState,
  loadChatHistory,
  loadAllChatsHistory,
  loadRecentMessages,
  preloadImportantChats,
  enhancedInitialSync
};
